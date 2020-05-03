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
! real*8,allocatable,dimension(:) :: ml_input_layer, ml_first_layer, ml_second_layer, ml_third_layer, ml_output_layer
! real*8,allocatable,dimension(:) :: ml_tmp
real*8 :: ml_sum, ml_weight, ml_biasweight, rotation
real*8 :: pi = 4 * atan (1.0_8)  ! Define pi
integer:: stz = 7
integer:: stzh = (7-1)/2
integer:: gridsize = 128
integer :: counter, w_idx_beg, b_idx_beg, i_loc, j_loc, n, o, p, counter1
#include "hahihjhk.h"
!####################################################################################################################################
! Calculate gradient of concentration field for pre-processing
call calgradc(ngr,0,1,c,dscdx,dscdy,dscdz, &
#               include "listcalgradcc.h"
        ier)  

! Allocate layers
! allocate(ml_input_layer(ml_layer_nodes(1)))
! allocate(ml_first_layer(ml_layer_nodes(2)))
! allocate(ml_second_layer(ml_layer_nodes(3)))
! allocate(ml_third_layer(ml_layer_nodes(4)))
! allocate(ml_output_layer(ml_layer_nodes(5)))
! allocate(ml_tmp(ml_layer_nodes(1)))

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
            ! .AND. ((c(ha(i+1,j  ,k)) > 0.01) .AND. (c(ha(i+1,j  ,k)) < 0.99))&
            ! .AND. ((c(ha(i-1,j  ,k)) > 0.01) .AND. (c(ha(i-1,j  ,k)) < 0.99))&
            ! .AND. ((c(ha(i  ,j+1,k)) > 0.01) .AND. (c(ha(i  ,j+1,k)) < 0.99))&
            ! .AND. ((c(ha(i  ,j-1,k)) > 0.01) .AND. (c(ha(i  ,j-1,k)) < 0.99))&
            ) then

          ! MACHINE LEARNING

          do counter=1,ml_layer_nodes(1)
            ml_input = 0
            ml_tmp = 0
          end do
          ! Fill input layer (ml_last_layer_output) with vof concentration values
          counter = 1
          do j_loc=-stzh,stzh
            do i_loc=-stzh,stzh
              ml_input_layer(counter) = cm(ha(i+i_loc, j+j_loc, 1))
              counter = counter+1
            end do
          end do
          
          ! PRE-PROCESSING
          ! Calculate quadrant of normal vector 
          ! 0 = upper left
          ! 1 = upper right
          ! 2 = lower right
          ! 3 = lower left
          rotation = floor(((atan(dscdy(inp), dscdx(inp))+pi)*1/(2*pi))*4)
          ! Rotate stencil so the top right corner is always 1 and the bottom left corner always 0
          counter = 1
          if (rotation == 1) then ! do not rotate
            do i_loc=1,stz**2
              ml_tmp(counter) = ml_input_layer(counter)
              counter = counter+1
            end do
          else if (rotation == 2) then ! rotate stencil by 90 degrees
            do j_loc=1,stz
              do i_loc=1,stz
                ml_tmp((stz+1-i_loc)*stz+(j_loc-stz)) = ml_input_layer(counter)
                counter = counter+1
              end do
            end do
          else if (rotation == 3) then ! rotate stencil by 180 degrees
            do i_loc=1,stz**2
              ml_tmp(counter) = ml_input_layer(stz**2+1-i_loc)
              counter = counter+1
            end do
          else if (rotation == 0) then ! rotate stencil by 270 degrees
            do j_loc=1,stz
              do i_loc=1,stz
                ml_tmp((i_loc-1)*stz+(stz+1-j_loc)) = ml_input_layer(counter)
                counter = counter+1
              end do
            end do
          end if
          
          ! Overwrite NN features with rotated stencils
          do i_loc=1,stz**2
            ml_input_layer(i_loc) = ml_tmp(i_loc)
          end do

          ! Export c/cm
          if (1 == 0) then
            print*, 'c'
            do j_loc=-stzh,stzh
              do i_loc=-stzh,stzh
                print*, c(ha(i+i_loc, j+j_loc, 1))
              end do
            end do
            print*, 'cm'
            ! do j_loc=-stzh,stzh
              ! do i_loc=-stzh,stzh
                ! print*, cm(ha(i+i_loc, j+j_loc, 1))
              ! end do
            ! end do
            do i_loc=1,stz**2  ! export rotated cm
              print*, ml_input_layer(i_loc)
            end do
          end if

          ! Hidden Layers

          ! First layer
          n = 2
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
              ml_sum = ml_sum+ml_input_layer(p)*ml_weight
            end do 
            ml_biasweight = ml_biasweights(b_idx_beg+o)
            ml_first_layer(o) = ml_sum+ml_biasweight
          end do

          ! Apply activation function (only for relu, do nothing if activation is linear)
          if (ml_layer_activation(n) == 'relu') then
            do o=1,ml_layer_nodes(n)
              ! Set every output that is smaller then 0 to 0
              if (ml_first_layer(o) < 0) then
                ml_first_layer(o) = 0
              end if
            end do
          end if



          ! Second layer
          n = 3

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
              ml_sum = ml_sum+ml_first_layer(p)*ml_weight
            end do 
            ml_biasweight = ml_biasweights(b_idx_beg+o)
            ml_second_layer(o) = ml_sum+ml_biasweight
          end do

          ! Apply activation function (only for relu, do nothing if activation is linear)
          if (ml_layer_activation(n) == 'relu') then
            do o=1,ml_layer_nodes(n)
              ! Set every output that is smaller then 0 to 0
              if (ml_second_layer(o) < 0) then
                ml_second_layer(o) = 0
              end if
            end do
          end if

          ! Third layer
          n = 4

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
              ml_sum = ml_sum+ml_second_layer(p)*ml_weight
            end do 
            ml_biasweight = ml_biasweights(b_idx_beg+o)
            ml_third_layer(o) = ml_sum+ml_biasweight
          end do

          ! Apply activation function (only for relu, do nothing if activation is linear)
          if (ml_layer_activation(n) == 'relu') then
            do o=1,ml_layer_nodes(n)
              ! Set every output that is smaller then 0 to 0
              if (ml_third_layer(o) < 0) then
                ml_third_layer(o) = 0
              end if
            end do
          end if


          ! Output Layer
          n = 5

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
              ml_sum = ml_sum+ml_third_layer(p)*ml_weight
            end do 
            ml_biasweight = ml_biasweights(b_idx_beg+o)
            ml_output_layer(o) = ml_sum+ml_biasweight
          end do

          ! Apply activation function (only for relu, do nothing if activation is linear)
          if (ml_layer_activation(n) == 'relu') then
            do o=1,ml_layer_nodes(n)
              ! Set every output that is smaller then 0 to 0
              if (ml_output_layer(o) < 0) then
                ml_output_layer(o) = 0
              end if
            end do
          end if

          dscurv2(inp)=-1*ml_output_layer(1)*0.5*gridsize/8

        else
          ! If concentration in stencil midpoint is 0 or 1, set curvature to 0
          dscurv2(inp)=0
        end if
      end do
    end do
  end do
end do
! Am Ende muss dscdx und dscurv2 (KrÃmmung) Ã¼ber dem ganzen StencilÃ¼berschrieben werden
! Das sind globale Variablen, die nicht ausgegeben werden mÃssen
end subroutine curv_ml
