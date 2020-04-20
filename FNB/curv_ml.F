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
integer :: i,j,k,m,inp,incst
real*8,allocatable,dimension(:) :: ml_last_layer_output
real*8,allocatable,dimension(:) :: ml_layer_input
real*8 :: ml_sum, ml_weight, ml_biasweight
integer :: counter, w_idx_beg, b_idx_beg, i_loc, j_loc, n, o, p, counter1
#include "hahihjhk.h"
!####################################################################################################################################


do m=1,nblo
  call setind(ngr,m, &
#             include "listsetind.h"
             ier)
  do k=2,nkm
    do j=2,njm
      do i=2,nim
        inp=ha(i,j,k)
        ! Only continue if vof concentration in stencil midpoint (and neighbors) is not 0 or 1
        if (((c(ha(i  ,j  ,k)) > 0.01) .AND. (c(ha(i  ,j  ,k)) < 0.99)) &
            .AND. ((c(ha(i+1,j  ,k)) > 0.01) .AND. (c(ha(i+1,j  ,k)) < 0.99))&
            .AND. ((c(ha(i-1,j  ,k)) > 0.01) .AND. (c(ha(i-1,j  ,k)) < 0.99))&
            .AND. ((c(ha(i  ,j+1,k)) > 0.01) .AND. (c(ha(i  ,j+1,k)) < 0.99))&
            .AND. ((c(ha(i  ,j-1,k)) > 0.01) .AND. (c(ha(i  ,j-1,k)) < 0.99))&
            ) then

          ! MACHINE LEARNING
          ! Input Layer
          allocate(ml_last_layer_output(ml_layer_nodes(1)))
          do counter=1,ml_layer_nodes(1)
            ml_last_layer_output = 0
          end do
          counter = 1
          ! Fill input layer (ml_last_layer_output) with vof concentration values
          ! WIE RUM MUSS DAS HIER SEIN DAMIT ES MIT DEM MACHINE LEARNING UEBEREINSTIMMT?
          ! PRE-PROCESSING MUSS AUCH NOCH REIN
          do j_loc=-3,3
            do i_loc=-3,3
              ml_last_layer_output(counter) = cm(ha(i+i_loc, j+j_loc, 1))
              counter = counter+1
            end do
          end do

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
          dscurv2(inp)=ml_last_layer_output(1)
          ! print*, 'size last layer', size(ml_last_layer_output)

          ! if (((ml_last_layer_output(1) > -4.1167E-002) .AND. (ml_last_layer_output(1) < -4.1164E-002)) &
            ! .OR. ((ml_last_layer_output(1) > -2.198E-002) .AND. (ml_last_layer_output(1) < -2.197E-002)) &
          ! ) then
            ! Export c/cm
          if (1 == 2) then
            print*, 'c'
            do j_loc=-3,3
              do i_loc=-3,3
                print*, c(ha(i+i_loc, j+j_loc, 1))
              end do
            end do
            print*, 'cm'
            do j_loc=-3,3
              do i_loc=-3,3
                print*, cm(ha(i+i_loc, j+j_loc, 1))
              end do
            end do
            print*, 'last output', ml_last_layer_output(1)
          end if

          deallocate(ml_last_layer_output)
        else
          ! If concentration in stencil midpoint is 0 or 1, set curvature to 0
          dscurv2(inp)=0
        end if
      end do
    end do
  end do
end do
call calgradc(ngr,0,1,c,dscdx,dscdy,dscdz, &
#               include "listcalgradcc.h"
        ier)  
! Am Ende muss dscdx und dscurv2 (KrÃmmung) Ã¼ber dem ganzen StencilÃ¼berschrieben werden
! Das sind globale Variablen, die nicht ausgegeben werden mÃssen
end subroutine curv_ml
