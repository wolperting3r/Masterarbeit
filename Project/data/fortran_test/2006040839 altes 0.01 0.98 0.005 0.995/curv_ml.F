#include "listdefines.h"
!##################################################################################################
subroutine curv_ml(ngr, &
#                  include "listcalcurv.h"
                   ier)
!##################################################################################################
! Height-Functions-Method
!##################################################################################################
use ml_mod
implicit none
#include "cb3dall.h"
#include "cbglobaldim.h"

! dummy variables
integer, intent(in out) :: ngr
#include "dimbcoef.h"
#include "dimgeom12.h"
#include "dimdscurv.h"
#include "dimdsst.h"
#include "dimconc.h"
#include "dimlilk.h"
#include "dimconcold.h"
#include "dimdivers.h"
#include "dimindex2.h"
#include "dimindex4.h"
#include "dimliglkg.h"
#include "dimlogic4.h"
#include "dimdiscf.h"
#include "dimtayint.h"
#include "dimrhelp3.h"
#include "dimbndcon.h"
#include "dimblopar.h"
#include "dimacoef.h"
#include "dimdsls.h"
#include "dimdslog.h"
#include "tradef.h"
#include "dimiters.h"
#include "dimcderivatives.h"

integer :: ier
      
! local variables
integer :: i,j,k,m,inp !,incst
real*8,allocatable,dimension(:) :: ml_last_layer_output
real*8,allocatable,dimension(:) :: ml_layer_input
real*8,allocatable,dimension(:) :: ml_tmp
real*8 :: ml_sum, ml_weight, ml_biasweight, rotation, ml_flip_sum_first, ml_flip_sum_last
real*8 :: xm, ym
real*8 :: pi = 4 * atan (1.0_8)  ! Define pi
integer:: stz
integer:: stzh
integer:: checksum
integer:: gridsize = 128
integer :: counter, w_idx_beg, b_idx_beg, i_loc, j_loc, n, o, p, counter1, i_shift, j_shift
integer :: export = 0

integer :: icphi
! integer,intent(in)  :: ngr
! real*8, intent (out):: dscurv2a
      
logical :: lfy
! local variables
logical :: laux,lfend,curvfiltering,curvweightening,secondweighting,flatsurfacectrl,nolocalcurv
! integer :: i,j,k,l,m,count1,inp,checkit,zerocount
!real*8  :: vz,xm,ym,zm,h,gh,dscurvsum2
integer :: curvfilter,weightsexp,weightsexp2
real*8  :: tempcurv(nxyza),tempcurv2(nxyza),weights(nxyza),normalized
real*8  :: fi1(nxyza),fi2(nxyza),fi3(nxyza),fi4(nxyza)
integer :: k1,kn,i1,inn,j1,jn,kstart,kend,numa,numb,incst,incst1
! real*8  :: xclim1,xclim2,xclim1_3d,xclim2_3d,scalingfactor
! real*4  :: t1, delta
#include "hahihjhk.h"
!####################################################################################################################################
! Calculate gradient of concentration field for pre-processing
call calgradc(ngr,0,1,c,dscdx,dscdy,dscdz, &
#               include "listcalgradcc.h"
        ier)  


stz = sqrt(real(ml_n_features))
stzh = (stz-1)/2

! t1 = secnds(0.0)
if ((it > 3200) .and. (it < 3500)) then
 export = 0
else 
 export = 0
end if


do m=1,nblo
  call setind(ngr,m, &
#             include "listsetind.h"
             ier)
  do k=2,nkm
    do j=2,njm
      do i=2,nim
        inp=ha(i,j,k)

        xm = sqrt((x(ha(i, j, k)) - x(ha(i-1, j, k)))**2)
        ym = sqrt((y(ha(i, j, k)) - y(ha(i, j-1, k)))**2)

        ! Check if points around ijk satisfy the condition
        checksum = 0
        !if ((cm(ha(i+1,j  ,k)) > 0.02) .AND. (cm(ha(i+1,j  ,k)) < 0.98)) then
          !checksum = checksum + 1
        !end if
        !if ((cm(ha(i,j-1,k)) > 0.02) .AND. (cm(ha(i,j-1,k)) < 0.98)) then
          !checksum = checksum + 1
        !end if
        !if ((cm(ha(i,j+1,k)) > 0.02) .AND. (cm(ha(i,j+1,k)) < 0.98)) then
          !checksum = checksum + 1
        !end if
        !if ((cm(ha(i-1,j  ,k)) > 0.02) .AND. (cm(ha(i-1,j  ,k)) < 0.98)) then
          !checksum = checksum + 1
        !end if
        !if ((cm(ha(i+1,j+1,k)) > 0.02) .AND. (cm(ha(i+1,j+1,k)) < 0.98)) then
          !checksum = checksum + 1
        !end if
        !if ((cm(ha(i+1,j-1,k)) > 0.02) .AND. (cm(ha(i+1,j-1,k)) < 0.98)) then
          !checksum = checksum + 1
        !end if
        !if ((cm(ha(i-1,j+1,k)) > 0.02) .AND. (cm(ha(i-1,j+1,k)) < 0.98)) then
          !checksum = checksum + 1
        !end if
        !if ((cm(ha(i-1,j-1,k)) > 0.02) .AND. (cm(ha(i-1,j-1,k)) < 0.98)) then
          !checksum = checksum + 1
        !end if

        ! Only continue if vof concentration in stencil midpoint (and neighbors) is not 0 or 1
        if (((cm(ha(i  ,j  ,k)) > 0.01) .AND. (cm(ha(i  ,j  ,k)) < 0.98)) &  ! war bei den Test meistens 0.98
        !if (((cm(ha(i  ,j  ,k)) > 0.01) .AND. (cm(ha(i  ,j  ,k)) < 0.99)) &  ! war bei den Test meistens 0.98
            !.AND. ((sqrt(dscdx(inp)**2 + dscdy(inp)**2) > 0.01d0/min(xm, ym))) & ! zusätzliche Bedingung
            !.AND. (checksum >= 2) &
            !.AND. (((c(ha(i+1,j  ,k)) > 0.075) .AND. (c(ha(i+1,j  ,k)) < 0.99))&
            !.OR. ((c(ha(i-1,j  ,k)) > 0.075) .AND. (c(ha(i-1,j  ,k)) < 0.99))&
            !.OR. ((c(ha(i  ,j+1,k)) > 0.075) .AND. (c(ha(i  ,j+1,k)) < 0.99))&
            !.OR. ((c(ha(i  ,j-1,k)) > 0.075) .AND. (c(ha(i  ,j-1,k)) < 0.99)))&
            ) then

          ! MACHINE LEARNING
          ! Input Layer
          allocate(ml_last_layer_output(ml_layer_nodes(1)))
          allocate(ml_tmp(ml_layer_nodes(1)))
          do counter=1,ml_layer_nodes(1)
            ml_last_layer_output = 0
            ml_tmp = 0
          end do
          ! Fill input layer (ml_last_layer_output) with vof concentration values
 
          counter = 1
          do j_loc=-stzh,stzh
            do i_loc=-stzh,stzh
              ! Set blocks where c < 0.005 to 0
              if (cm(ha(i+i_loc, j+j_loc, 1)) < 0.005) then
                ml_last_layer_output(counter) = 0
              else if (cm(ha(i+i_loc, j+j_loc, 1)) > 0.995) then
                ml_last_layer_output(counter) = 1
              else
                ml_last_layer_output(counter) = cm(ha(i+i_loc, j+j_loc, 1))
              end if
              counter = counter+1
            end do
          end do

          if (1 == 1) then  ! rotation switch
            ! Calculate quadrant of normal vector 
            ! 0 = upper left (old)/
            ! 1 = upper right (old)/
            ! 2 = lower right (old)/
            ! 3 = lower left (old)/
            ! Rotate gradient by 45 degrees first
            ! print*, 'Gradient'
            ! print*, dscdx(inp), dscdy(inp)
            rotation = floor(((atan(-dscdy(inp), -dscdx(inp))+pi)*1/(2*pi))*8)  ! Old version
            ! Rotate stencil so the top right corner is always 1 and the bottom left corner always 0
            counter = 1

            if (rotation == 0) then  ! war 7
              do j_loc=1,stz
                do i_loc=1,stz
                  ml_tmp(i_loc + (j_loc-1)*stz) = ml_last_layer_output(counter)
                  counter = counter+1
                end do
              end do
            else if (rotation == 1) then  ! war 6
              do j_loc=1,stz
                do i_loc=1,stz
                  ml_tmp(j_loc + (i_loc -1)*stz) = ml_last_layer_output(counter)
                  counter = counter+1
                end do
              end do
            else if (rotation == 6) then  ! war 5
              do j_loc=1,stz
                do i_loc=1,stz
                  ml_tmp((i_loc-1)*stz + (stz+1-j_loc)) = ml_last_layer_output(counter)
                  counter = counter+1
                end do
              end do
            else if (rotation == 3) then  ! war 4
              do j_loc=1,stz
                do i_loc=1,stz
                  ml_tmp((j_loc-1)*stz + (stz+1-i_loc)) = ml_last_layer_output(counter)
                  counter = counter+1
                end do
              end do
            else if (rotation == 4) then  ! war 3
              do j_loc=1,stz
                do i_loc=1,stz
                  ml_tmp(stz*stz+1-i_loc - (j_loc-1)*stz) = ml_last_layer_output(counter)
                  counter = counter+1
                end do
              end do
            else if (rotation == 5) then  ! war 2
              do j_loc=1,stz
                do i_loc=1,stz
                  ml_tmp(stz*stz+1-j_loc-(i_loc-1)*stz) = ml_last_layer_output(counter)
                  counter = counter+1
                end do
              end do
            else if (rotation == 2) then  ! war 1
              do j_loc=1,stz
                do i_loc=1,stz
                  ml_tmp((stz+1-i_loc)*stz+(j_loc-stz)) = ml_last_layer_output(counter)
                  counter = counter+1
                end do
              end do
            else if (rotation == 7) then  ! war 0
              do j_loc=1,stz
                do i_loc=1,stz
                  ml_tmp((stz+1-j_loc)*stz+(i_loc-stz)) = ml_last_layer_output(counter)
                  counter = counter+1
                end do
              end do
            end if


            ! Overwrite NN features with rotated stencils
            do i_loc=1,stz**2
              ml_last_layer_output(i_loc) = ml_tmp(i_loc)
            end do

            deallocate(ml_tmp)
          end if

          ! Export c/cm
          if (export == 1) then
            print*, 'c'
            do j_loc=-stzh,stzh
              do i_loc=-stzh,stzh
                print*, cm(ha(i+i_loc, j+j_loc, 1))
              end do
            end do
            print*, 'cm'
            ! do j_loc=-stzh,stzh
              ! do i_loc=-stzh,stzh
                ! print*, cm(ha(i+i_loc, j+j_loc, 1))
              ! end do
            ! end do
            do i_loc=1,stz**2  ! export rotated cm
              print*, ml_last_layer_output(i_loc)
            end do
          end if

          ! ml_last_layer_output = (/0.0, 0.0, 0.0, 0.00166667, 0.11166667, 0.69, 0.97416667,&
          ! 0.0, 0.0, 0.0, 0.0425, 0.44, 0.93166667, 1.0, 0.0, 0.0,&
          ! 0.0075, 0.17666667, 0.80916667, 0.99166667, 1.0, 0.0,&
          ! 0.0, 0.04916667, 0.5, 0.95, 1.0, 1.0, 0.0, 0.0075,&
          ! 0.17416667, 0.8225, 0.9925, 1.0, 1.0, 0.0, 0.04,&
          ! 0.46583333, 0.9475, 1.0, 1.0, 1.0, 0.00166667, 0.1025,&
          ! 0.75333333, 0.985, 1.0, 1.0, 1.0/)

          ! Hidden Layers
          do n=2,ml_n_layers
            ! # ml_layer_input will be the input of the current layer, ml_last_layer_ouput will be used to save the results for the next layer
            ! Set ml_layer_input = ml_last_layer_output
            ! DAS IST WAHRSCHEINLICH LANGSAM
            allocate(ml_layer_input(size(ml_last_layer_output)))
            do counter=1,size(ml_last_layer_output)
              ml_layer_input(counter) = 0
            end do
            do counter=1,size(ml_layer_input)
              ml_layer_input(counter)=ml_last_layer_output(counter)
            end do
            ! Reallocate ml_last_layer_output to be filled with values of current layer
            deallocate(ml_last_layer_output)
            allocate(ml_last_layer_output(ml_layer_nodes(n)))
            do counter=1,ml_layer_nodes(n)
              ml_last_layer_output(counter) = 0
            end do

            ! Get index for weights and bias vector where current layer starts by summing length of weights of all previous layers
            w_idx_beg = 0
            do counter1=1,(n-1)
              w_idx_beg = w_idx_beg+ml_n_layerweights(counter1)
            end do
            b_idx_beg = 0
            do counter1=1,(n-1)
              b_idx_beg = b_idx_beg+ml_n_biasweights(counter1)
            end do

            ! Calculate current layer input*weights + bias
            do o=1,ml_layer_nodes(n)
              ml_sum = 0
              ml_weight = 0
              ! Sum all connections (output of last layer l *weight k l) for node k of current layer
              do p=1,ml_layer_nodes(n-1)
                ! Get weight of connection between last layer and current layer
                ml_weight = ml_layerweights(w_idx_beg+(p-1)*ml_layer_nodes(n)+o)
                ! Add influence of that connection
                ml_sum = ml_sum+ml_layer_input(p)*ml_weight
              end do 
              ml_biasweight = ml_biasweights(b_idx_beg+o)
              ml_last_layer_output(o) = ml_sum+ml_biasweight
            end do

            ! Apply activation function (only for relu, do nothing if activation is linear)
            if (ml_layer_activation(n) == 'relu') then
              do o=1,ml_layer_nodes(n)
                ! Set every output that is smaller then 0 to 0
                if (ml_last_layer_output(o) < 0) then
                  ml_last_layer_output(o) = 0
                end if
              end do
            end if

            ! Deallocate input for usage in next layer
            deallocate(ml_layer_input)
          end do
          ! Last output of ML network is the curvature 
          ! dscurv2(inp)=-1*ml_last_layer_output(1)*0.5*gridsize/8
          dscurv2(inp)=-1*ml_last_layer_output(1)*0.5*gridsize/0.075

          ! print*, dscurv2(inp)
          ! dscurv2(inp) = 0.5
          if (export == 1) then
            print*, 'last output', dscurv2(inp)
            !print*, 'last output', rotation
          endif
          ! print*, 'size last layer', size(ml_last_layer_output)

          ! if (((ml_last_layer_output(1) > -4.1167E-002) .AND. (ml_last_layer_output(1) < -4.1164E-002)) &
            ! .OR. ((ml_last_layer_output(1) > -2.198E-002) .AND. (ml_last_layer_output(1) < -2.197E-002)) &
          ! ) then

          deallocate(ml_last_layer_output)
        else
          ! If concentration in stencil midpoint is 0 or 1, set curvature to 0
          dscurv2(inp)=0
        end if
      end do
    end do
  end do
end do
!STOP

curvfiltering=.true.  ! war false
 !1-lsp0,2-binominal,3-5pointcheby,4-lsp2
curvfilter=3
curvweightening=.true.
weightsexp=1

! Filter the curvature afterwards
if (curvfiltering .eqv. .true.) then

  !first step
  call setzero(weights,nxyza,1,nxyza,ier)
  call setzero(tempcurv,nxyza,1,nxyza,ier)
  call exall(ngr,dscurv2,1,1, &
#            include "listexall.h"
             ier)
  do icphi=0,mcphi
    incst=icphi*nxyza
    incst1=incst+1
    call exall(ngr,cm(incst1),1,1, &
#            include "listexall.h"
             ier)
  end do
  do m=1,nblo
    call setind(ngr,m, &
#               include "listsetind.h"
                ier)
    if (thirddim==1) then
      kstart=1
      kend=nk
    else
      kstart=2
      kend=2
    end if
    do k=kstart,kend
      do i=1,ni
        do j=1,nj
          inp=ha(i,j,k)
          ! more weight to cells with 0.5 concentration, less close to 0 or 1
          ! if ((cm(inp) > 0.01) .AND. (cm(inp) < 0.98)) then
          if (dscurv2(inp)/=0d0) then
            weights(inp)=1
          else
            weights(inp)=0
          end if
        end do
      end do
    end do
  end do
  do m=1,nblo
    call setind(ngr,m, &
#               include "listsetind.h"
                ier)
    do k=2,nkm
      do i=2,nim
        do j=2,njm
          inp=ha(i,j,k)
          if (dscurv2(inp)/=0d0) then
            if (thirddim==0) then
              tempcurv(inp)=(dscurv2(inp)*4*weights(inp) + &
                !dscurv2(ha(i+1,j+1,k))*weights(ha(i+1,j+1,k))+ &  ! von mir
                !dscurv2(ha(i+1,j-1,k))*weights(ha(i+1,j-1,k))+ &  ! von mir
                !dscurv2(ha(i-1,j+1,k))*weights(ha(i-1,j+1,k))+ &  ! von mir
                !dscurv2(ha(i-1,j-1,k))*weights(ha(i-1,j-1,k))+ &  ! von mir
                dscurv2(ha(i+1,j,k))*weights(ha(i+1,j,k))+ &
                dscurv2(ha(i-1,j,k))*weights(ha(i-1,j,k))+ &
                dscurv2(ha(i,j+1,k))*weights(ha(i,j+1,k))+ &
                dscurv2(ha(i,j-1,k))*weights(ha(i,j-1,k)))/ &
                !(weights(inp)+weights(ha(i+1,j,k))+weights(ha(i+1,j+1,k))+weights(ha(i+1,j-1,k))+&
                !weights(ha(i-1,j+1,k))+weights(ha(i-1,j-1,k))+weights(ha(i-1,j,k))+weights(ha(i,j+1,k))+weights(ha(i,j-1,k)))
                (weights(inp)*4+weights(ha(i+1,j,k))+weights(ha(i-1,j,k))+weights(ha(i,j+1,k))+weights(ha(i,j-1,k)))  ! vor meiner Aenderung
            else
              tempcurv(inp)=(dscurv2(inp)*weights(inp) + &
                dscurv2(ha(i+1,j,k))*weights(ha(i+1,j,k))+ &
                dscurv2(ha(i-1,j,k))*weights(ha(i-1,j,k))+ &
                dscurv2(ha(i,j+1,k))*weights(ha(i,j+1,k))+ &
                dscurv2(ha(i,j-1,k))*weights(ha(i,j-1,k))+ &
                dscurv2(ha(i,j,k+1))*weights(ha(i,j,k+1))+ &
                dscurv2(ha(i,j,k-1))*weights(ha(i,j,k-1)))/ &
                (weights(inp)+weights(ha(i+1,j,k))+weights(ha(i-1,j,k))+weights(ha(i,j+1,k))+weights(ha(i,j-1,k)) &
                +weights(ha(i,j,k+1))+weights(ha(i,j,k-1)))
            end if
          end if
        end do
      end do
    end do
  end do
  do m=1,nblo
    call setind(ngr,m, &
#               include "listsetind.h"
                ier)
    do k=2,nkm
      do i=2,nim
        do j=2,njm
          inp=ha(i,j,k)
          if (tempcurv(inp)/=0d0) then
            dscurv2(inp)=tempcurv(inp)
          end if
        end do
      end do
    end do
  end do
end if
! Weightening of curvature
if (curvweightening .eqv. .true.) then

  !first step
  call setzero(weights,nxyza,1,nxyza,ier)
  call setzero(tempcurv,nxyza,1,nxyza,ier)
  call exall(ngr,dscurv2,1,1, &
#            include "listexall.h"
             ier)
  do icphi=0,mcphi
    incst=icphi*nxyza
    incst1=incst+1
    call exall(ngr,cm(incst1),1,1, &
#            include "listexall.h"
             ier)
  end do
  do m=1,nblo
    call setind(ngr,m, &
#               include "listsetind.h"
                ier)
    if (thirddim==1) then
      kstart=1
      kend=nk
    else
      kstart=2
      kend=2
    end if
    do k=kstart,kend
      do i=1,ni
        do j=1,nj
          inp=ha(i,j,k)
          ! more weight to cells with 0.5 concentration, less close to 0 or 1
          ! if ((cm(inp) > 0.01) .AND. (cm(inp) < 0.98)) then
          if (dscurv2(inp)/=0d0) then
            weights(inp)=(1-2d0*abs(0.5d0-cm(inp)))**(weightsexp)
          else
            weights(inp)=0
          end if
        end do
      end do
    end do
  end do
  do m=1,nblo
    call setind(ngr,m, &
#               include "listsetind.h"
                ier)
    do k=2,nkm
      do i=2,nim
        do j=2,njm
          inp=ha(i,j,k)
          ! if (dscurv2(inp)/=0d0) then
          if ((cm(inp) > 0.005) .AND. (cm(inp) < 0.995)) then
            if (thirddim==0) then
              tempcurv(inp)=(dscurv2(inp)*weights(inp) + &
                dscurv2(ha(i+1,j+1,k))*weights(ha(i+1,j+1,k))+ &  ! von mir
                dscurv2(ha(i+1,j-1,k))*weights(ha(i+1,j-1,k))+ &  ! von mir
                dscurv2(ha(i-1,j+1,k))*weights(ha(i-1,j+1,k))+ &  ! von mir
                dscurv2(ha(i-1,j-1,k))*weights(ha(i-1,j-1,k))+ &  ! von mir
                dscurv2(ha(i+1,j,k))*weights(ha(i+1,j,k))+ &
                dscurv2(ha(i-1,j,k))*weights(ha(i-1,j,k))+ &
                dscurv2(ha(i,j+1,k))*weights(ha(i,j+1,k))+ &
                dscurv2(ha(i,j-1,k))*weights(ha(i,j-1,k)))/ &
                (weights(inp)+weights(ha(i+1,j,k))+weights(ha(i+1,j+1,k))+weights(ha(i+1,j-1,k))+&
                weights(ha(i-1,j+1,k))+weights(ha(i-1,j-1,k))+weights(ha(i-1,j,k))+weights(ha(i,j+1,k))+weights(ha(i,j-1,k)))
                !(weights(inp)+weights(ha(i+1,j,k))+weights(ha(i-1,j,k))+weights(ha(i,j+1,k))+weights(ha(i,j-1,k)))  ! vor meiner Aenderung
            else
              tempcurv(inp)=(dscurv2(inp)*weights(inp) + &
                dscurv2(ha(i+1,j,k))*weights(ha(i+1,j,k))+ &
                dscurv2(ha(i-1,j,k))*weights(ha(i-1,j,k))+ &
                dscurv2(ha(i,j+1,k))*weights(ha(i,j+1,k))+ &
                dscurv2(ha(i,j-1,k))*weights(ha(i,j-1,k))+ &
                dscurv2(ha(i,j,k+1))*weights(ha(i,j,k+1))+ &
                dscurv2(ha(i,j,k-1))*weights(ha(i,j,k-1)))/ &
                (weights(inp)+weights(ha(i+1,j,k))+weights(ha(i-1,j,k))+weights(ha(i,j+1,k))+weights(ha(i,j-1,k)) &
                +weights(ha(i,j,k+1))+weights(ha(i,j,k-1)))
            end if
          end if
        end do
      end do
    end do
  end do
  do m=1,nblo
    call setind(ngr,m, &
#               include "listsetind.h"
                ier)
    do k=2,nkm
      do i=2,nim
        do j=2,njm
          inp=ha(i,j,k)
          if (tempcurv(inp)/=0d0) then
            dscurv2(inp)=tempcurv(inp)
          end if
        end do
      end do
    end do
  end do
end if


! delta = secnds(t1)
! print*, delta
! Am Ende muss dscdx und dscurv2 (KrÃmmung) Ã¼ber dem ganzen StencilÃ¼berschrieben werden
! Das sind globale Variablen, die nicht ausgegeben werden mÃssen
end subroutine curv_ml
