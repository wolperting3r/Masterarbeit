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
integer :: ml_loc
real*8 :: ml_sum_rc, ml_mean, ml_mean2
real*8 :: xm, ym
real*8 :: pi = 4 * atan (1.0_8)  ! Define pi
integer, dimension(1, 1) :: ml_case
integer:: stz
integer:: stzh
integer:: checksum
integer:: gridsize = 128
integer :: counter, w_idx_beg, b_idx_beg, i_loc, j_loc, n, o, p, counter1, i_shift, j_shift
integer :: j_locc, i_locc ! Am Ende lÃschen
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

        if (((cm(ha(i  ,j  ,k)) > 0.05d0) .AND. (cm(ha(i  ,j  ,k)) < 0.95d0)) &  ! war bei den Test meistens 0.98
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
          do counter=1,ml_layer_nodes(1)
            ml_last_layer_output(counter) = 0
          end do
          ! Fill input layer (ml_last_layer_output) with vof concentration values
 
          counter = 1
          do j_loc=-stzh,stzh
            do i_loc=-stzh,stzh
              ! Set blocks where c < 0.005 to 0
              ! if (c(ha(i+i_loc, j+j_loc, 1)) < 0.005) then
                ! ml_last_layer_output(counter) = 0
              ! else if (c(ha(i+i_loc, j+j_loc, 1)) > 0.995) then
                ! ml_last_layer_output(counter) = 1
              ! else
              ml_last_layer_output(counter) = cm(ha(i+i_loc, j+j_loc, 1))
              ! end if
              counter = counter+1
            end do
          end do

          if (1 == 1) then  ! rotation switch
            allocate(ml_tmp(ml_layer_nodes(1)))
            do counter=1,ml_layer_nodes(1)
              ml_tmp = 0
            end do
            ! Calculate quadrant of normal vector 
            ! 0 = upper left (old)/
            ! 1 = upper right (old)/
            ! 2 = lower right (old)/
            ! 3 = lower left (old)/
            ! Rotate gradient by 45 degrees first
            ! print*, 'Gradient'
            ! print*, dscdx(inp), dscdy(inp)
            rotation = floor(((atan(-dscdy(inp), -dscdx(inp))+pi)*1d0/(2d0*pi))*8d0)  ! Old version
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


          if (ml_edge_on == 1) then  ! edge switch

              ! Test
              ! deallocate(ml_last_layer_output)
              ! allocate(ml_last_layer_output(81))

              ! ml_last_layer_output = (/&
                ! 1.0,  1.0,  0.4, 1.0,  0.2,  0.2,  0.2,  0.0, 0.0,&
                ! 1.0,  1.0,  0.9, 1.0,  0.6,  0.45, 0.45, 0.0, 0.0,&
                ! 1.0,  1.0,  1.0, 1.0,  1.0,  0.9,  0.9,  0.2, 0.9,&
                ! 1.0,  1.0,  1.0, 1.0,  1.0,  1.0,  0.85, 0.6, 0.4,&
                ! 1.0,  1.0,  1.0, 1.0,  0.75, 0.75, 0.45, 0.1, 0.9,&
                ! 1.0,  1.0,  0.9, 0.8,  0.55, 0.55, 0.12, 0.0, 0.0,&
                ! 1.0,  0.8,  0.7, 0.1,  0.4,  0.4,  0.01, 0.0, 0.0,&
                ! 1.0,  0.601,0.3, 0.05, 0.0,  0.0,  0.0,  0.0, 0.0,&
                ! 1.0,  0.4,  0.2, 0.01, 0.0,  0.0,  0.0,  0.0, 0.0&
              ! /)  
              ! ml_last_layer_output = (/&
                ! 1.0,  1.0,   1.0,  1.0,  1.0,  1.0,  1.0, 1.0,  1.0,&
                ! 0.4,  0.601, 0.8,  1.0,  1.0,  1.0,  1.0, 1.0,  1.0,&
                ! 0.2,  0.3,   0.7,  0.9,  1.0,  1.0,  1.0, 0.9,  0.4,&
                ! 0.01, 0.05,  0.1,  0.8,  1.0,  1.0,  1.0, 1.0,  1.0,&
                ! 0.0,  0.0,   0.4,  0.55, 0.75, 1.0,  1.0, 0.6,  0.2,&
                ! 0.0,  0.0,   0.4,  0.55, 0.75, 1.0,  0.9, 0.45, 0.2,&
                ! 0.0,  0.0,   0.01, 0.12, 0.45, 0.85, 0.9, 0.45, 0.2,&
                ! 0.0,  0.0,   0.0,  0.0,  0.1,  0.6,  0.2, 0.0,  0.0,&
                ! 0.0,  0.0,   0.0,  0.0,  0.9,  0.4,  0.9, 0.0,  0.0&
              ! /)

              ! stz = 9
              ! stzh = 4
              ! allocate(ml_mask(stz, stz))
              ! allocate(ml_mask_x((stz-2), (stz-2)))
              ! allocate(ml_mask_y((stz-2), (stz-2)))
              ! allocate(ml_and_mask_x((stz-2), (stz-3)))
              ! allocate(ml_and_mask_y((stz-3), (stz-2)))
              ! allocate(ml_sum_mask_x((stz-2)))
              ! allocate(ml_sum_mask_y((stz-2)))
              ! allocate(ml_sum_and_mask_x((stz-3)))
              ! allocate(ml_sum_and_mask_y((stz-3)))

              ! allocate(ml_interp_x((stz-2), (stz-2)))
              ! allocate(ml_interp_y((stz-2), (stz-2)))
              ! allocate(ml_data_cut((stz-2), (stz-2)))
              ! allocate(ml_mask_cut((stz-2), (stz-2)))
              ! allocate(ml_data_x((stz-2), (stz-2)))
              ! allocate(ml_data_y((stz-2), (stz-2)))
              ! allocate(ml_row((stz-2)))
              ! allocate(ml_and_row((stz-3)))

            do j_loc=1,stz
              do i_loc=1,stz
                ml_mask(j_loc, i_loc) = 0
              end do
            end do

            do j_loc=1,stz-2
              do i_loc=1,stz-2
                ml_mask_x(j_loc, i_loc) = 0
                ml_mask_y(j_loc, i_loc) = 0
              end do
            end do
            do j_loc=1,stz-3
              do i_loc=1,stz-2
                ml_and_mask_y(j_loc, i_loc) = 0
              end do
            end do
            do j_loc=1,stz-2
              do i_loc=1,stz-3
                ml_and_mask_y(j_loc, i_loc) = 0
              end do
            end do

            counter = 1
            ! Get mask: c > 0.5 -> 1, c < 0.5 -> 0
            do j_loc=1,stz
              do i_loc=1,stz
                if (ml_last_layer_output(counter) >= 0.5d0) then
                  ml_mask(j_loc, i_loc) = 1
                else
                  ml_mask(j_loc, i_loc) = 0
                end if

                if (((j_loc > 1) .and. (j_loc < stz)) .and. ((i_loc > 1) .and. (i_loc < stz))) then
                  ! Get inner part of stencil
                  ml_data_cut(j_loc-1, i_loc-1) = ml_last_layer_output(counter)
                  ml_mask_cut(j_loc-1, i_loc-1) = ml_mask(j_loc, i_loc)
                end if

                counter = counter+1
              end do
            end do


            ! Get mask_x, mask_y
            do j_loc=2,stz-1
              do i_loc=2,stz-1
                ml_mask_x(j_loc-1, i_loc-1) = or(&
                xor(ml_mask(j_loc, i_loc), ml_mask(j_loc, i_loc-1)),&
                xor(ml_mask(j_loc, i_loc), ml_mask(j_loc, i_loc+1)))
              end do
            end do

            do j_loc=2,stz-1
              do i_loc=2,stz-1
                ml_mask_y(j_loc-1, i_loc-1) = or(&
                xor(ml_mask(j_loc, i_loc), ml_mask(j_loc-1, i_loc)),&
                xor(ml_mask(j_loc, i_loc), ml_mask(j_loc+1, i_loc)))
              end do
            end do

            ! Get and_mask_x, and_mask_y
            do j_loc=1,stz-2
              do i_loc=1,stz-3
                ml_and_mask_x(j_loc, i_loc) = and(ml_mask_x(j_loc, i_loc), ml_mask_x(j_loc, i_loc+1))
              end do
            end do

            do j_loc=1,stz-3
              do i_loc=1,stz-2
                ml_and_mask_y(j_loc, i_loc) = and(ml_mask_y(j_loc, i_loc), ml_mask_y(j_loc+1, i_loc))
              end do
            end do

            ! Sum over mask_x/mask_y
            do j_loc=1,stz-2
              ml_sum_mask_x(j_loc) = 0
              do i_loc=1,stz-2
                ml_sum_mask_x(j_loc) = ml_sum_mask_x(j_loc)+ml_mask_x(j_loc, i_loc)
              end do
            end do
            do i_loc=1,stz-2
              ml_sum_mask_y(i_loc) = 0
              do j_loc=1,stz-2
                ml_sum_mask_y(i_loc) = ml_sum_mask_y(i_loc)+ml_mask_y(j_loc, i_loc)
              end do
            end do

            ! Sum over and_mask_x/and_mask_y
            do j_loc=1,stz-2
              ml_sum_and_mask_x(j_loc) = 0
              do i_loc=1,stz-3
                ml_sum_and_mask_x(j_loc) = ml_sum_and_mask_x(j_loc)+ml_and_mask_x(j_loc, i_loc)
              end do
            end do
            do i_loc=1,stz-2
              ml_sum_and_mask_y(i_loc) = 0
              do j_loc=1,stz-3
                ml_sum_and_mask_y(i_loc) = ml_sum_and_mask_y(i_loc)+ml_and_mask_y(j_loc, i_loc)
              end do
            end do

            ! Data_x/data_y
            do j_loc=1,stz-2
              do i_loc=1,stz-2
                ml_data_x(j_loc, i_loc) = ml_mask_x(j_loc, i_loc)*ml_data_cut(j_loc, i_loc)
                ml_data_y(j_loc, i_loc) = ml_mask_y(j_loc, i_loc)*ml_data_cut(j_loc, i_loc)
              end do
            end do

            ! Ueber Reihen (x) bzw. Spalten (y) iterieren
            ! Fallunterscheidung
            ! Und gleich die Reihe/Spalte bearbeiten
            do j_loc=1,stz-2
              do i_loc=1,stz-2
                ml_interp_x(j_loc, i_loc) = 0
                ml_interp_y(j_loc, i_loc) = 0
              end do
            end do
            
            ! Iterate over rows for x
            do j_loc=1,stz-2
              ! Get row
              do i_loc=1,stz-2
                ml_row(i_loc) = ml_data_x(j_loc, i_loc)
              end do
              do i_loc=1,stz-3
                ml_and_row(i_loc) = ml_and_mask_x(j_loc, i_loc)
              end do

              ! print"(f6.3, f6.3, f6.3, f6.3, f6.3, f6.3, f6.3, f6.3, f6.3)",&
                ! ml_last_layer_output(j_loc*stz+1), ml_last_layer_output(j_loc*stz+2), ml_last_layer_output(j_loc*stz+3),&
                ! ml_last_layer_output(j_loc*stz+4), ml_last_layer_output(j_loc*stz+5), ml_last_layer_output(j_loc*stz+6),&
                ! ml_last_layer_output(j_loc*stz+7), ml_last_layer_output(j_loc*stz+8), ml_last_layer_output(j_loc*stz+9)
              ! print"(f12.3, f6.3, f6.3, f6.3, f6.3, f6.3, f6.3)", ml_row
              ! print"(i6.1, i6.1, i6.1, i6.1, i6.1, i6.1, i6.1, i6.1, i6.1)", ml_mask(j_loc+1, 1),&
              ! ml_mask(j_loc+1, 2), ml_mask(j_loc+1, 3), ml_mask(j_loc+1, 4), ml_mask(j_loc+1, 5),&
              ! ml_mask(j_loc+1, 6), ml_mask(j_loc+1, 7), ml_mask(j_loc+1, 8), ml_mask(j_loc+1, 9)
              ! print"(i12.1, i6.1, i6.1, i6.1, i6.1, i6.1, i6.1)", ml_mask_x(j_loc, 1),&
              ! ml_mask_x(j_loc, 2), ml_mask_x(j_loc, 3), ml_mask_x(j_loc, 4), ml_mask_x(j_loc, 5),&
              ! ml_mask_x(j_loc, 6), ml_mask_x(j_loc, 7)
              ! print"(i12.1, i6.1, i6.1, i6.1, i6.1, i6.1)", ml_and_mask_x(j_loc, 1),&
              ! ml_and_mask_x(j_loc, 2), ml_and_mask_x(j_loc, 3), ml_and_mask_x(j_loc, 4), ml_and_mask_x(j_loc, 5),&
              ! ml_and_mask_x(j_loc, 6)
              ! print"(f6.3)", ml_sum_mask_x(j_loc)
              ! print"(f6.3)", ml_sum_and_mask_x(j_loc)
              ! print*, ' '

              
              ! Decide case
              if ((ml_sum_mask_x(j_loc) == 1) .and. (ml_sum_and_mask_x(j_loc) == 0)) then
                ! One single (1 0 0 0 0)
                ! Just copy row to output stencil
                do i_loc=1,stz-2
                  ml_interp_x(j_loc, i_loc) = ml_row(i_loc)
                end do

              else if ((ml_sum_mask_x(j_loc) == 2) .and. (ml_sum_and_mask_x(j_loc) == 0)) then
                ! Two single (1 0 0 0 1)
                ! Just copy row to output stencil
                do i_loc=1,stz-2
                  ml_interp_x(j_loc, i_loc) = ml_row(i_loc)
                end do

              else if ((ml_sum_mask_x(j_loc) == 2) .and. (ml_sum_and_mask_x(j_loc) == 1)) then
                ! One pair (0 1 1 0 0 0)
                ! Take mean of row and copy (+/-0.5) on bigger if < 0.5, smaller if > 0.5
                ml_sum_rc = 0
                do i_loc=1,stz-2
                  ml_sum_rc = ml_sum_rc + ml_row(i_loc)
                end do

                ml_mean = ml_sum_rc/2

                do i_loc=1,stz-2
                  ml_interp_x(j_loc, i_loc) = 0
                end do
                if (ml_mean > 0.5d0) then
                  ! Write ml_mean-0.5 on smaller value
                  ml_interp_x(j_loc, minloc(ml_row, mask=(ml_row>0))) = ml_mean-0.5d0
                else
                  ! Write ml_mean+0.5 on bigger value
                  ml_interp_x(j_loc, maxloc(ml_row)) = ml_mean+0.5d0
                end if

              else if ((ml_sum_mask_x(j_loc) == 3) .and. (ml_sum_and_mask_x(j_loc) == 1)) then
                ! One pair, one single (1 1 0 0 1)
                ! Get mean of pair
                ml_mean = ((ml_row(maxloc(ml_and_row, DIM=1)) + ml_row((maxloc(ml_and_row, DIM=1)+1)))/2)
                do i_loc=1,stz-2
                  ml_interp_x(j_loc, i_loc) = 0
                end do
                if (ml_mean > 0.5d0) then
                  ! Write ml_mean-0.5 on smaller value
                  if (ml_row(maxloc(ml_and_row, DIM=1)) .ge. ml_row(maxloc(ml_and_row, DIM=1)+1)) then
                    ! maxloc(ml_and_row) is bigger, so write on maxloc(ml_and_row)+1
                    ml_interp_x(j_loc, (maxloc(ml_and_row, DIM=1)+1)) = ml_mean-0.5d0
                  else
                    ! maxloc(ml_and_row) is smaller, so write on maxloc(ml_and_row)
                    ml_interp_x(j_loc, (maxloc(ml_and_row, DIM=1))) = ml_mean-0.5d0
                  end if
                else
                  ! Write ml_mean+0.5 on bigger value
                  if (ml_row(maxloc(ml_and_row, DIM=1)) .ge. ml_row(maxloc(ml_and_row, DIM=1)+1)) then
                    ! maxloc(ml_and_row) is bigger, so write on maxloc(ml_and_row)
                    ml_interp_x(j_loc, (maxloc(ml_and_row, DIM=1))) = ml_mean+0.5d0
                  else
                    ! maxloc(ml_and_row) is smaller, so write on maxloc(ml_and_row)+1
                    ml_interp_x(j_loc, (maxloc(ml_and_row, DIM=1)+1)) = ml_mean+0.5d0
                  end if
                end if

                ! Set values of pair in row to zero so only single is left
                ml_row(maxloc(ml_and_row, DIM=1)) = 0
                ml_row(maxloc(ml_and_row, DIM=1)+1) = 0
                ! Copy single
                ml_interp_x(j_loc, maxloc(ml_row, DIM=1)) = ml_row(maxloc(ml_row, DIM=1))

              else if ((ml_sum_mask_x(j_loc) == 3) .and. (ml_sum_and_mask_x(j_loc) == 2)) then
                ! One triple (1 1 1 0 0)
                ! and          1 1 0 0

                ! Get first index of triplet (maxloc returns first max value)
                ml_loc = maxloc(ml_and_row, DIM=1)

                ml_mean = (ml_row(ml_loc) + ml_row(ml_loc+1))/2
                ml_mean2 = (ml_row(ml_loc+1) + ml_row(ml_loc+2))/2

                if (ml_mean > 0.5d0) then
                  ml_mean = ml_mean - 0.5d0
                else
                  ml_mean = ml_mean + 0.5d0
                end if

                if (ml_mean2 > 0.5d0) then
                  ml_mean2 = ml_mean2 - 0.5d0
                else
                  ml_mean2 = ml_mean2 + 0.5d0
                end if

                do i_loc=1,stz-2
                  ml_interp_x(j_loc, i_loc) = 0
                end do

                if (and( xor((ml_mean > 0.5d0), (ml_row(ml_loc) > ml_row(ml_loc+1))),&
                  xor((ml_mean2 > 0.5d0), (ml_row(ml_loc+2) > ml_row(ml_loc+1))))) then
                  ! Clash -> mean of means on inner
                  ml_interp_x(j_loc, (ml_loc+1)) = (ml_mean+ml_mean2)/2
                else
                  ! No clash
                  if (xor((ml_mean > 0.5d0), (ml_row(ml_loc) > ml_row(ml_loc+1)))) then
                    ! Write ml_mean on inner
                    ml_interp_x(j_loc, (ml_loc+1)) = ml_mean
                  else
                    ! Write ml_mean on outer
                    ml_interp_x(j_loc, ml_loc) = ml_mean
                  end if

                  if (xor((ml_mean2 > 0.5d0), (ml_row(ml_loc+2) > ml_row(ml_loc+1)))) then
                    ! Write ml_mean2 on inner
                    ml_interp_x(j_loc, (ml_loc+1)) = ml_mean2
                  else
                    ! Write ml_mean2 on outer
                    ml_interp_x(j_loc, (ml_loc+2)) = ml_mean2
                  end if
                end if

              else if ((ml_sum_mask_x(j_loc) == 4) .and. (ml_sum_and_mask_x(j_loc) == 2)) then
                ! Two pairs, separated (1 1 0 1 1)
                ! Get first index of first pair (maxloc returns first max value)
                ml_loc = maxloc(ml_and_row, DIM=1)
                ml_mean = (ml_row(ml_loc) + ml_row(ml_loc+1))/2

                do i_loc=1,stz-2
                  ml_interp_x(j_loc, i_loc) = 0
                end do
                if (ml_mean > 0.5d0) then
                  ! Write ml_mean-0.5 on smaller value
                  if (ml_row(ml_loc) > ml_row(ml_loc+1)) then
                    ! ml_loc > ml_loc + 1, so write on ml_loc+1
                    ml_interp_x(j_loc, ml_loc+1) = ml_mean-0.5d0
                  else
                    ! ml_loc <= ml_loc + 1, so write on ml_loc
                    ml_interp_x(j_loc, ml_loc) = ml_mean-0.5d0
                  end if
                else
                  ! Write ml_mean+0.5 on bigger value
                  if (ml_row(ml_loc) > ml_row(ml_loc+1)) then
                    ! ml_loc > ml_loc + 1, so write on ml_loc
                    ml_interp_x(j_loc, ml_loc) = ml_mean+0.5d0
                  else
                    ! ml_loc <= ml_loc + 1, so write on ml_loc+1
                    ml_interp_x(j_loc, ml_loc+1) = ml_mean+0.5d0
                  end if
                end if

                ! Set values in row to 0
                ml_row(ml_loc) = 0
                ml_row(ml_loc+1) = 0
                ml_and_row(ml_loc) = 0

                ! And repeat for second pair
                ml_loc = maxloc(ml_and_row, DIM=1)
                ml_mean = (ml_row(ml_loc) + ml_row(ml_loc+1))/2

                if (ml_mean > 0.5d0) then
                  ! Write ml_mean-0.5 on smaller value
                  if (ml_row(ml_loc) > ml_row(ml_loc+1)) then
                    ! ml_loc > ml_loc + 1, so write on ml_loc+1
                    ml_interp_x(j_loc, ml_loc+1) = ml_mean-0.5d0
                  else
                    ! ml_loc <= ml_loc + 1, so write on ml_loc
                    ml_interp_x(j_loc, ml_loc) = ml_mean-0.5d0
                  end if
                else
                  ! Write ml_mean+0.5 on bigger value
                  if (ml_row(ml_loc) > ml_row(ml_loc+1)) then
                    ! ml_loc > ml_loc + 1, so write on ml_loc
                    ml_interp_x(j_loc, ml_loc) = ml_mean+0.5d0
                  else
                    ! ml_loc <= ml_loc + 1, so write on ml_loc+1
                    ml_interp_x(j_loc, ml_loc+1) = ml_mean+0.5d0
                  end if
                end if

              else if ((ml_sum_mask_x(j_loc) == 4) .and. (ml_sum_and_mask_x(j_loc) == 3)) then
                ! Two pairs, combined (1 1 1 1 0)

                ! Get first index of first pair (maxloc returns first max value)
                ml_loc = maxloc(ml_and_row, DIM=1)
                ml_mean = (ml_row(ml_loc) + ml_row(ml_loc+1))/2

                do i_loc=1,stz-2
                  ml_interp_x(j_loc, i_loc) = 0
                end do
                if (ml_mean > 0.5d0) then
                  ! Write ml_mean-0.5 on smaller value
                  if (ml_row(ml_loc) > ml_row(ml_loc+1)) then
                    ! ml_loc > ml_loc + 1, so write on ml_loc+1
                    ml_interp_x(j_loc, ml_loc+1) = ml_mean-0.5d0
                  else
                    ! ml_loc <= ml_loc + 1, so write on ml_loc
                    ml_interp_x(j_loc, ml_loc) = ml_mean-0.5d0
                  end if
                else
                  ! Write ml_mean+0.5 on bigger value
                  if (ml_row(ml_loc) > ml_row(ml_loc+1)) then
                    ! ml_loc > ml_loc + 1, so write on ml_loc
                    ml_interp_x(j_loc, ml_loc) = ml_mean+0.5d0
                  else
                    ! ml_loc <= ml_loc + 1, so write on ml_loc+1
                    ml_interp_x(j_loc, ml_loc+1) = ml_mean+0.5d0
                  end if
                end if

                ! Set values in row to 0
                ml_row(ml_loc) = 0
                ml_row(ml_loc+1) = 0
                ml_and_row(ml_loc) = 0
                ml_and_row(ml_loc+1) = 0

                ! And repeat for second pair
                ml_loc = maxloc(ml_and_row, DIM=1)
                ml_mean = (ml_row(ml_loc) + ml_row(ml_loc+1))/2

                if (ml_mean > 0.5d0) then
                  ! Write ml_mean-0.5 on smaller value
                  if (ml_row(ml_loc) > ml_row(ml_loc+1)) then
                    ! ml_loc > ml_loc + 1, so write on ml_loc+1
                    ml_interp_x(j_loc, ml_loc+1) = ml_mean-0.5d0
                  else
                    ! ml_loc <= ml_loc + 1, so write on ml_loc
                    ml_interp_x(j_loc, ml_loc) = ml_mean-0.5d0
                  end if
                else
                  ! Write ml_mean+0.5 on bigger value
                  if (ml_row(ml_loc) > ml_row(ml_loc+1)) then
                    ! ml_loc > ml_loc + 1, so write on ml_loc
                    ml_interp_x(j_loc, ml_loc) = ml_mean+0.5d0
                  else
                    ! ml_loc <= ml_loc + 1, so write on ml_loc+1
                    ml_interp_x(j_loc, ml_loc+1) = ml_mean+0.5d0
                  end if
                end if

              end if
            end do

            ! Iterate over columns for y
            do i_loc=1,stz-2
              ! Get column
              do j_loc=1,stz-2
                ml_row(j_loc) = ml_data_y(j_loc, i_loc)
              end do
              do j_loc=1,stz-3
                ml_and_row(j_loc) = ml_and_mask_y(j_loc, i_loc)
              end do
              
              ! Decide case
              if ((ml_sum_mask_y(i_loc) == 1) .and. (ml_sum_and_mask_y(i_loc) == 0)) then
                ! One single (1 0 0 0 0)
                ! Just copy column to output stencil
                do j_loc=1,stz-2
                  ml_interp_y(j_loc, i_loc) = ml_data_y(j_loc, i_loc)
                end do

              else if ((ml_sum_mask_y(i_loc) == 2) .and. (ml_sum_and_mask_y(i_loc) == 0)) then
                ! Two single (1 0 0 0 1)
                ! Just copy column to output stencil
                do j_loc=1,stz-2
                  ml_interp_y(j_loc, i_loc) = ml_data_y(j_loc, i_loc)
                end do

              else if ((ml_sum_mask_y(i_loc) == 2) .and. (ml_sum_and_mask_y(i_loc) == 1)) then
                ! One pair (0 1 1 0 0 0)
                ! Take mean of row and copy (+/-0.5) on bigger if < 0.5, smaller if > 0.5
                ml_sum_rc = 0
                do j_loc=1,stz-2
                  ml_sum_rc = ml_sum_rc + ml_row(j_loc)
                end do

                ml_mean = ml_sum_rc/2

                do j_loc=1,stz-2
                  ml_interp_y(j_loc, i_loc) = 0
                end do
                if (ml_mean > 0.5d0) then
                  ! Write ml_mean-0.5 on smaller value
                  ml_interp_y(minloc(ml_row, mask=(ml_row>0)), i_loc) = ml_mean-0.5d0
                else
                  ! Write ml_mean+0.5 on bigger value
                  ml_interp_y(maxloc(ml_row), i_loc) = ml_mean+0.5d0
                end if

              else if ((ml_sum_mask_y(i_loc) == 3) .and. (ml_sum_and_mask_y(i_loc) == 1)) then
                ! One pair, one single (1 1 0 0 1)
                ! Get mean of pair
                ml_mean = ((ml_row(maxloc(ml_and_row, DIM=1)) + ml_row((maxloc(ml_and_row, DIM=1)+1)))/2)
                do j_loc=1,stz-2
                  ml_interp_y(j_loc, i_loc) = 0
                end do
                if (ml_mean > 0.5d0) then
                  ! Write ml_mean-0.5d0 on smaller value
                  if (ml_row(maxloc(ml_and_row, DIM=1)) .ge. ml_row(maxloc(ml_and_row, DIM=1)+1)) then
                    ! maxloc(ml_and_row) is bigger, so write on maxloc(ml_and_row)+1
                    ml_interp_y((maxloc(ml_and_row, DIM=1)+1), i_loc) = ml_mean-0.5d0
                  else
                    ! maxloc(ml_and_row) is smaller, so write on maxloc(ml_and_row)
                    ml_interp_y((maxloc(ml_and_row, DIM=1)), i_loc) = ml_mean-0.5d0
                  end if
                else
                  ! Write ml_mean+0.5d0 on bigger value
                  if (ml_row(maxloc(ml_and_row, DIM=1)) .ge. ml_row(maxloc(ml_and_row, DIM=1)+1)) then
                    ! maxloc(ml_and_row) is bigger, so write on maxloc(ml_and_row)
                    ml_interp_y((maxloc(ml_and_row, DIM=1)), i_loc) = ml_mean+0.5d0
                  else
                    ! maxloc(ml_and_row) is smaller, so write on maxloc(ml_and_row)+1
                    ml_interp_y((maxloc(ml_and_row, DIM=1)+1), i_loc) = ml_mean+0.5d0
                  end if
                end if

                ! Set values of pair in row to zero so only single is left
                ml_row(maxloc(ml_and_row, DIM=1)) = 0
                ml_row(maxloc(ml_and_row, DIM=1)+1) = 0
                ! Copy single
                ml_interp_y(maxloc(ml_row, DIM=1), i_loc) = ml_row(maxloc(ml_row, DIM=1))

              else if ((ml_sum_mask_y(i_loc) == 3) .and. (ml_sum_and_mask_y(i_loc) == 2)) then
                ! One triple (1 1 1 0 0)
                ! and          1 1 0 0

                ! Get first index of triplet (maxloc returns first max value)
                ml_loc = maxloc(ml_and_row, DIM=1)

                ml_mean = (ml_row(ml_loc) + ml_row(ml_loc+1))/2
                ml_mean2 = (ml_row(ml_loc+1) + ml_row(ml_loc+2))/2

                if (ml_mean > 0.5d0) then
                  ml_mean = ml_mean - 0.5d0
                else
                  ml_mean = ml_mean + 0.5d0
                end if

                if (ml_mean2 > 0.5d0) then
                  ml_mean2 = ml_mean2 - 0.5d0
                else
                  ml_mean2 = ml_mean2 + 0.5d0
                end if

                do j_loc=1,stz-2
                  ml_interp_y(j_loc, i_loc) = 0
                end do

                if (and( xor((ml_mean > 0.5d0), (ml_row(ml_loc) > ml_row(ml_loc+1))),&
                  xor((ml_mean2 > 0.5d0), (ml_row(ml_loc+2) > ml_row(ml_loc+1))))) then
                  ! Clash -> mean of means
                  ml_interp_y((ml_loc+1), i_loc) = (ml_mean+ml_mean2)/2
                else
                  ! No clash
                  if (xor((ml_mean > 0.5d0), (ml_row(ml_loc) > ml_row(ml_loc+1)))) then
                    ! Write ml_mean on inner
                    ml_interp_y((ml_loc+1), i_loc) = ml_mean
                  else
                    ! Write ml_mean on outer
                    ml_interp_y(ml_loc, i_loc) = ml_mean
                  end if

                  if (xor((ml_mean2 > 0.5d0), (ml_row(ml_loc+2) > ml_row(ml_loc+1)))) then
                    ! Write ml_mean2 on inner
                    ml_interp_y((ml_loc+1), i_loc) = ml_mean2
                  else
                    ! Write ml_mean2 on outer
                    ml_interp_y((ml_loc+2), i_loc) = ml_mean2
                  end if
                end if

              else if ((ml_sum_mask_y(i_loc) == 4) .and. (ml_sum_and_mask_y(i_loc) == 2)) then
                ! Two pairs, separated (1 1 0 1 1)
                ! Get first index of first pair (maxloc returns first max value)
                ml_loc = maxloc(ml_and_row, DIM=1)
                ml_mean = (ml_row(ml_loc) + ml_row(ml_loc+1))/2

                do j_loc=1,stz-2
                  ml_interp_y(j_loc, i_loc) = 0
                end do
                if (ml_mean > 0.5d0) then
                  ! Write ml_mean-0.5d0 on smaller value
                  if (ml_row(ml_loc) > ml_row(ml_loc+1)) then
                    ! ml_loc > ml_loc + 1, so write on ml_loc+1
                    ml_interp_y(ml_loc+1, i_loc) = ml_mean-0.5d0
                  else
                    ! ml_loc <= ml_loc + 1, so write on ml_loc
                    ml_interp_y(ml_loc, i_loc) = ml_mean-0.5d0
                  end if
                else
                  ! Write ml_mean+0.5d0 on bigger value
                  if (ml_row(ml_loc) > ml_row(ml_loc+1)) then
                    ! ml_loc > ml_loc + 1, so write on ml_loc
                    ml_interp_y(ml_loc, i_loc) = ml_mean+0.5d0
                  else
                    ! ml_loc <= ml_loc + 1, so write on ml_loc+1
                    ml_interp_y(ml_loc+1, i_loc) = ml_mean+0.5d0
                  end if
                end if

                ! Set values in row to 0
                ml_row(ml_loc) = 0
                ml_row(ml_loc+1) = 0
                ml_and_row(ml_loc) = 0

                ! And repeat for second pair
                ml_loc = maxloc(ml_and_row, DIM=1)
                ml_mean = (ml_row(ml_loc) + ml_row(ml_loc+1))/2

                if (ml_mean > 0.5) then
                  ! Write ml_mean-0.5 on smaller value
                  if (ml_row(ml_loc) > ml_row(ml_loc+1)) then
                    ! ml_loc > ml_loc + 1, so write on ml_loc+1
                    ml_interp_y(ml_loc+1, i_loc) = ml_mean-0.5d0
                  else
                    ! ml_loc <= ml_loc + 1, so write on ml_loc
                    ml_interp_y(ml_loc, i_loc) = ml_mean-0.5d0
                  end if
                else
                  ! Write ml_mean+0.5d0 on bigger value
                  if (ml_row(ml_loc) > ml_row(ml_loc+1)) then
                    ! ml_loc > ml_loc + 1, so write on ml_loc
                    ml_interp_y(ml_loc, i_loc) = ml_mean+0.5d0
                  else
                    ! ml_loc <= ml_loc + 1, so write on ml_loc+1
                    ml_interp_y(ml_loc+1, i_loc) = ml_mean+0.5d0
                  end if
                end if

              else if ((ml_sum_mask_y(i_loc) == 4) .and. (ml_sum_and_mask_y(i_loc) == 3)) then
                ! Two pairs, combined (1 1 1 1 0)

                ! Get first index of first pair (maxloc returns first max value)
                ml_loc = maxloc(ml_and_row, DIM=1)
                ml_mean = (ml_row(ml_loc) + ml_row(ml_loc+1))/2

                do j_loc=1,stz-2
                  ml_interp_y(j_loc, i_loc) = 0
                end do
                if (ml_mean > 0.5d0) then
                  ! Write ml_mean-0.5d0 on smaller value
                  if (ml_row(ml_loc) > ml_row(ml_loc+1)) then
                    ! ml_loc > ml_loc + 1, so write on ml_loc+1
                    ml_interp_y(ml_loc+1, i_loc) = ml_mean-0.5d0
                  else
                    ! ml_loc <= ml_loc + 1, so write on ml_loc
                    ml_interp_y(ml_loc, i_loc) = ml_mean-0.5d0
                  end if
                else
                  ! Write ml_mean+0.5d0 on bigger value
                  if (ml_row(ml_loc) > ml_row(ml_loc+1)) then
                    ! ml_loc > ml_loc + 1, so write on ml_loc
                    ml_interp_y(ml_loc, i_loc) = ml_mean+0.5d0
                  else
                    ! ml_loc <= ml_loc + 1, so write on ml_loc+1
                    ml_interp_y(ml_loc+1, i_loc) = ml_mean+0.5d0
                  end if
                end if

                ! Set values in row to 0
                ml_row(ml_loc) = 0
                ml_row(ml_loc+1) = 0
                ml_and_row(ml_loc) = 0
                ml_and_row(ml_loc+1) = 0

                ! And repeat for second pair
                ml_loc = maxloc(ml_and_row, DIM=1)
                ml_mean = (ml_row(ml_loc) + ml_row(ml_loc+1))/2

                if (ml_mean > 0.5d0) then
                  ! Write ml_mean-0.5d0 on smaller value
                  if (ml_row(ml_loc) > ml_row(ml_loc+1)) then
                    ! ml_loc > ml_loc + 1, so write on ml_loc+1
                    ml_interp_y(ml_loc+1, i_loc) = ml_mean-0.5d0
                  else
                    ! ml_loc <= ml_loc + 1, so write on ml_loc
                    ml_interp_y(ml_loc, i_loc) = ml_mean-0.5d0
                  end if
                else
                  ! Write ml_mean+0.5d0 on bigger value
                  if (ml_row(ml_loc) > ml_row(ml_loc+1)) then
                    ! ml_loc > ml_loc + 1, so write on ml_loc
                    ml_interp_y(ml_loc, i_loc) = ml_mean+0.5d0
                  else
                    ! ml_loc <= ml_loc + 1, so write on ml_loc+1
                    ml_interp_y(ml_loc+1, i_loc) = ml_mean+0.5d0
                  end if
                end if

              end if
            end do

            ! Fill with mask 
            do j_loc=1,stz-2
              do i_loc=1,stz-2
                ml_interp_x(j_loc, i_loc) = ml_interp_x(j_loc, i_loc) + ml_mask_cut(j_loc, i_loc)
                if (ml_interp_x(j_loc, i_loc) > 1) then
                  ml_interp_x(j_loc, i_loc) = ml_interp_x(j_loc, i_loc) -1
                end if

                ml_interp_y(j_loc, i_loc) = ml_interp_y(j_loc, i_loc) + ml_mask_cut(j_loc, i_loc)
                if (ml_interp_y(j_loc, i_loc) > 1) then
                  ml_interp_y(j_loc, i_loc) = ml_interp_y(j_loc, i_loc) -1
                end if
              end do
            end do

            ! Test

            ! print*, ' '
            ! print"(1i2)", j_loc
            ! print*, ' '
            
            ! print"(f6.3, f6.3, f6.3, f6.3, f6.3, f6.3, f6.3)", ml_row
            ! print*, ' '
            ! print*, 'data_cut'
            ! do j_locc=1,stz-2
              ! print"(f6.3, f6.3, f6.3, f6.3, f6.3, f6.3, f6.3)", ml_data_cut(j_locc, 1), ml_data_cut(j_locc, 2),&
                ! ml_data_cut(j_locc, 3), ml_data_cut(j_locc, 4), ml_data_cut(j_locc, 5), ml_data_cut(j_locc, 6),&
                ! ml_data_cut(j_locc, 7)
            ! end do

            ! print*, 'interp_x'
            ! do j_locc=1,stz-2
              ! print"(f6.3, f6.3, f6.3, f6.3, f6.3, f6.3, f6.3)", ml_interp_x(j_locc, 1), ml_interp_x(j_locc, 2),&
                ! ml_interp_x(j_locc, 3), ml_interp_x(j_locc, 4), ml_interp_x(j_locc, 5), ml_interp_x(j_locc, 6),&
                ! ml_interp_x(j_locc, 7)
            ! end do

            ! print*, 'interp_y'
            ! do j_locc=1,stz-2
              ! print"(f6.3, f6.3, f6.3, f6.3, f6.3, f6.3, f6.3)", ml_interp_y(j_locc, 1), ml_interp_y(j_locc, 2),&
                ! ml_interp_y(j_locc, 3), ml_interp_y(j_locc, 4), ml_interp_y(j_locc, 5), ml_interp_y(j_locc, 6),&
                ! ml_interp_y(j_locc, 7)
            ! end do

            deallocate(ml_last_layer_output)
            allocate(ml_last_layer_output(2*(stz-2)**2))

            ! Flatten ml_inter_x/y and feed it into NN
            counter=1
            do j_loc=1,stz-2
              do i_loc=1,stz-2
                ml_last_layer_output(counter) = ml_interp_x(j_loc, i_loc)
                counter = counter + 1
              end do
            end do
            do j_loc=1,stz-2
              do i_loc=1,stz-2
                ml_last_layer_output(counter) = ml_interp_y(j_loc, i_loc)
                counter = counter + 1
              end do
            end do

            ! do i_loc=1,2*(stz-2)**2
            ! print"(f6.3)", ml_last_layer_output(i_loc)
            ! end do

          end if

          ! Export c/cm
          if (export == 1) then
            print*, 'c'
            ! do j_loc=1,stz-3
              ! print*, ml_and_row(j_loc)
            ! end do
            ! do j_loc=1,stz
              ! do i_loc=1,stz
                ! print*, c(j_loc, i_loc)
              ! end do
            ! end do

            ! print*, 'cm'
            ! do j_loc=1,stz-2
              ! print*, ml_sum_and_mask_x(j_loc)
            ! end do
            ! do j_loc=1,stz-2
              ! do i_loc=1,stz-3
                ! print*, ml_and_mask_x(j_loc, i_loc)
              ! end do
            ! end do
            do j_loc=-stzh,stzh
              do i_loc=-stzh,stzh
                print*, cm(ha(i+i_loc, j+j_loc, 1))
              end do
            end do
            print*, 'cm'
            ! do j_loc=1,stz
              ! do i_loc=1,stz
                ! print*, cm(j_loc, i_loc)
              ! end do
            ! end do
            do i_loc=1,2*((stz-2)**2)  ! export rotated cm
              print*, ml_last_layer_output(i_loc)
            end do
          end if

          ! STOP

          ! Hidden Layers
          do n=2,ml_n_layers
            ! # ml_layer_input will be the input of the current layer, ml_last_layer_ouput will be used to save the results for the next layer
            ! Set ml_layer_input = ml_last_layer_output
            ! DAS IST WAHRSCHEINLICH LANGSAM
            allocate(ml_layer_input(size(ml_last_layer_output)))
            do counter=1,size(ml_last_layer_output)
              ml_layer_input(counter) = 0d0
            end do
            do counter=1,size(ml_layer_input)
              ml_layer_input(counter)=ml_last_layer_output(counter)
            end do
            ! Reallocate ml_last_layer_output to be filled with values of current layer
            deallocate(ml_last_layer_output)
            allocate(ml_last_layer_output(ml_layer_nodes(n)))
            do counter=1,ml_layer_nodes(n)
              ml_last_layer_output(counter) = 0d0
            end do

            ! Get index for weights and bias vector where current layer starts by summing length of weights of all previous layers
            w_idx_beg = 0d0
            do counter1=1,(n-1)
              w_idx_beg = w_idx_beg+ml_n_layerweights(counter1)
            end do
            if (ml_bias_on == 1) then
              b_idx_beg = 0d0
              do counter1=1,(n-1)
                b_idx_beg = b_idx_beg+ml_n_biasweights(counter1)
              end do
            end if

            ! Calculate current layer input*weights + bias
            do o=1,ml_layer_nodes(n)
              ml_sum = 0d0
              ml_weight = 0d0
              ! Sum all connections (output of last layer l *weight k l) for node k of current layer
              do p=1,ml_layer_nodes(n-1)
                ! Get weight of connection between last layer and current layer
                ml_weight = ml_layerweights(w_idx_beg+(p-1)*ml_layer_nodes(n)+o)
                ! Add influence of that connection
                ml_sum = ml_sum+ml_layer_input(p)*ml_weight
              end do 
              if (ml_bias_on == 1) then
                ml_biasweight = ml_biasweights(b_idx_beg+o)
                ml_last_layer_output(o) = ml_sum+ml_biasweight
              else
                ml_last_layer_output(o) = ml_sum
              end if
            end do

            ! Apply activation function (only for relu, do nothing if activation is linear)
            if (ml_layer_activation(n) == 'relu') then
              do o=1,ml_layer_nodes(n)
                ! Set every output that is smaller then 0 to 0
                if (ml_last_layer_output(o) < 0d0) then
                  ml_last_layer_output(o) = 0d0
                end if
              end do
            end if

            ! Deallocate input for usage in next layer
            deallocate(ml_layer_input)
          end do
          ! Last output of ML network is the curvature 
          ! dscurv2(inp)=-1*ml_last_layer_output(1)*0.5*gridsize/8
          ! dscurv2(inp)=1*ml_last_layer_output(1)*0.5d0*gridsize/0.075d0
          dscurv2(inp)=1*ml_last_layer_output(1)*gridsize/0.075d0  ! CVOFLS

          ! print*, dscurv2(inp)
          ! dscurv2(inp) = 0.5
          if (export == 1) then
            ! print*, 'last output', dscurv2(inp)
            print*, 'last output', ml_last_layer_output(1)
            !print*, 'last output', rotation
          endif
          ! STOP
          ! print*, 'size last layer', size(ml_last_layer_output)

          ! if (((ml_last_layer_output(1) > -4.1167E-002) .AND. (ml_last_layer_output(1) < -4.1164E-002)) &
            ! .OR. ((ml_last_layer_output(1) > -2.198E-002) .AND. (ml_last_layer_output(1) < -2.197E-002)) &
          ! ) then

          deallocate(ml_last_layer_output)
        else
          ! If concentration in stencil midpoint is 0 or 1, set curvature to 0
          dscurv2(inp)=0d0
        end if
      end do
    end do
  end do
end do
! print*, 'STOP'
! STOP

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
            ! weights(inp)=1
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
          if (dscurv2(inp)/=0d0) then
            if (thirddim==0) then
              tempcurv(inp)=(dscurv2(inp)*weights(inp) + &
                !dscurv2(ha(i+1,j+1,k))*weights(ha(i+1,j+1,k))+ &  ! von mir
                !dscurv2(ha(i+1,j-1,k))*weights(ha(i+1,j-1,k))+ &  ! von mir
                !dscurv2(ha(i-1,j+1,k))*weights(ha(i-1,j+1,k))+ &  ! von mir
                !dscurv2(ha(i-1,j-1,k))*weights(ha(i-1,j-1,k))+ &  ! von mir
                dscurv2(ha(i+1,j,k))*weights(ha(i+1,j,k))+ &
                dscurv2(ha(i-1,j,k))*weights(ha(i-1,j,k))+ &
                dscurv2(ha(i,j+1,k))*weights(ha(i,j+1,k))+ &
                dscurv2(ha(i,j-1,k))*weights(ha(i,j-1,k)))/ &
                !(4*weights(inp)+weights(ha(i+1,j,k))+weights(ha(i+1,j+1,k))+weights(ha(i+1,j-1,k))+&
                !weights(ha(i-1,j+1,k))+weights(ha(i-1,j-1,k))+weights(ha(i-1,j,k))+weights(ha(i,j+1,k))+weights(ha(i,j-1,k)))
                (weights(inp)+weights(ha(i+1,j,k))+weights(ha(i-1,j,k))+weights(ha(i,j+1,k))+weights(ha(i,j-1,k)))  ! vor meiner Aenderung
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
            ! weights(inp)=(1-2d0*abs(0.5d0-cm(inp)))**(weightsexp)
            weights(inp)=1d0
          else
            weights(inp)=0d0
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
          if ((cm(inp) > 0.005d0) .AND. (cm(inp) < 0.995d0)) then
            if (thirddim==0) then
              ! tempcurv(inp)=(dscurv2(inp)*weights(inp)*4 + &
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
                ! (4*weights(inp)+weights(ha(i+1,j,k))+weights(ha(i-1,j,k))+weights(ha(i,j+1,k))+weights(ha(i,j-1,k)))  ! vor meiner Aenderung
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
          if ((tempcurv(inp)/=0d0)) then
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
