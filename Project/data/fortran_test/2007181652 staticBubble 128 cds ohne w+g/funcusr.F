#include "listdefines.h"
!#####################################################################
subroutine usrini(ngr,adapdt, &
#                 include "listall.h"  
                  ier)       
!#####################################################################
! User routine called once on every grid level before the simulation.
!#####################################################################
use adaptime_mod
use ml_mod
implicit none
#include "cb3dall.h"
#include "cbglobaldim.h"

! Import machine learning global variables

! dummy variables
! Variables of the adaptative time step
type(adaptime_typ), intent(in out) :: adapdt
integer, intent(in) :: ngr
#include "dimall.h"
integer :: ier

! local variables
integer :: i,j,k,m,inp,itimes
integer :: icphi,incst,incst1
real*8 :: fden,fden2,fvis,fvis2,r1

integer :: stat  ! status of file io
integer                                 :: ml_totalweightlength
integer                                 :: ml_totalbiaslength
integer                                 :: ml_linecounter_layer
integer                                 :: ml_linecounter_bias
integer                                 :: stz

! arguments
integer                             :: npts(3)

! local variables
real*8, dimension(:,:), allocatable :: gridpoints
real*8                              :: vertices(8,3)
real*8                              :: radius,center_x,center_y,center_z,radius1,radius2
real*8                              :: a000,ax00,a0y0,axy0,a00z,ax0z,a0yz,axyz
integer                             :: idiv,jdiv,kdiv,ipts
integer                             :: nidiv,njdiv,nkdiv
integer                             :: tnpts,tndiv,nptsInside
character(len=7)                    :: typeOfObject
#include "hahihjhk.h"
!#######################################################################
call setzero(c,nxyzac,1,nxyzac)
call setzero(cold,nxyzac,1,nxyzac)
!############### Initialize Machine Learning ########################

! Open file with weights
loop_counter = 1
open(unit=142,file='model_mlp_1000_200-150-120_7x7_rot_ellipse_flp_cut_dshift1_shift1_bia_int2_ned_ncl_g_s1',&
  action='read',status='old',iostat=stat)
if (stat == 0) then
        read(142,*) ml_n_features       ! number of input features
        read(142,*) ml_n_layers         ! number of layers (including input, including output)
        ! Input layer is a layer too
        ml_n_layers = ml_n_layers+1

        ! Allocate global variables 
        allocate(ml_input(ml_n_features))               ! input features
        allocate(ml_layer_nodes(ml_n_layers))           ! vector with number of nodes
        allocate(ml_layer_activation(ml_n_layers))      ! vector with activation function


        ! First layer is input layer
        ml_layer_nodes(1)=ml_n_features
        ! Read number of nodes per layer
        do i=2,ml_n_layers
          read(142,*) ml_layer_nodes(i)
        end do

        ! First layer has no activation function
        ml_layer_activation(1)='line'
        ! Read activation function per layer
        do i=2,ml_n_layers
          read(142,*) ml_layer_activation(i)
        end do
        
        ! Read bias switch
        read(142,*) ml_bias_on
        ! Read edge switch
        read(142,*) ml_edge_on
        ! read(142,*) ml_out_bias_on

        ! Allocate variables needed for edge
        if (ml_edge_on == 1) then
          stz = sqrt(real(ml_n_features))
          ! Two times (stz-2)^2 instead of one time stz^2
          ml_layer_nodes(1) = 2*(stz-2)**2

          allocate(ml_mask(stz, stz))
          allocate(ml_mask_x((stz-2), (stz-2)))
          allocate(ml_mask_y((stz-2), (stz-2)))
          allocate(ml_and_mask_x((stz-2), (stz-3)))
          allocate(ml_and_mask_y((stz-3), (stz-2)))
          allocate(ml_sum_mask_x((stz-2)))
          allocate(ml_sum_mask_y((stz-2)))
          allocate(ml_sum_and_mask_x((stz-3)))
          allocate(ml_sum_and_mask_y((stz-3)))

          allocate(ml_interp_x((stz-2), (stz-2)))
          allocate(ml_interp_y((stz-2), (stz-2)))
          allocate(ml_data_cut((stz-2), (stz-2)))
          allocate(ml_mask_cut((stz-2), (stz-2)))
          allocate(ml_data_x((stz-2), (stz-2)))
          allocate(ml_data_y((stz-2), (stz-2)))
          allocate(ml_row((stz-2)))
          allocate(ml_and_row((stz-3)))
        end if

        ! Calculate number of layer weights
        allocate(ml_n_layerweights(ml_n_layers))
        if (ml_bias_on == 1) then
          allocate(ml_n_biasweights(ml_n_layers))
        end if

        ! Input layer has no weights
        ml_n_layerweights(1) = 0
        ! Tensors between layers
        do i=2,ml_n_layers
          ml_n_layerweights(i) = ml_layer_nodes(i-1)*ml_layer_nodes(i)
        end do
        ! Calculate total length of node weights
        ml_totalweightlength = 0
        do i=1,ml_n_layers
          ml_totalweightlength = ml_n_layerweights(i) + ml_totalweightlength
        end do
        ! Allocate node weight arrays with total length
        allocate(ml_layerweights(ml_totalweightlength))

        ! Do the same for biases
        if (ml_bias_on == 1) then
          ! Input layer has no biases
          ml_n_biasweights(1) = 0
          ! Biases are added to each hidden layer
          do i=2,ml_n_layers
            ml_n_biasweights(i) = ml_layer_nodes(i)
          end do
          ! Calculate total length of bias weights
          ml_totalbiaslength = 0
          do i=1,ml_n_layers
            ml_totalbiaslength = ml_n_biasweights(i) + ml_totalbiaslength
          end do
          ! Allocate bias weight arrays with total length
          allocate(ml_biasweights(ml_totalbiaslength))
        end if

        ml_linecounter_layer = 0
        ml_linecounter_bias = 0
        do i=1,ml_n_layers  ! Iterate over every layer
          ! Read node weights of layer i
          do j=1,ml_n_layerweights(i)  ! Iterate over all node weights of layer i
            read(142,*) ml_layerweights(ml_linecounter_layer+j) 
          end do
          ml_linecounter_layer = ml_linecounter_layer + ml_n_layerweights(i)

          ! Read bias weights of layer i
          if (ml_bias_on == 1) then
            do j=1,ml_n_biasweights(i)
              read(142,*) ml_biasweights(ml_linecounter_bias+j) 
            end do
            ml_linecounter_bias = ml_linecounter_bias + ml_n_biasweights(i)
          end if

        end do

        ! Allocate layers
        ! allocate(ml_input_layer(ml_layer_nodes(1)))
        ! allocate(ml_first_layer(ml_layer_nodes(2)))
        ! allocate(ml_second_layer(ml_layer_nodes(3)))
        ! allocate(ml_third_layer(ml_layer_nodes(4)))
        ! allocate(ml_output_layer(ml_layer_nodes(5)))
        ! allocate(ml_tmp(ml_layer_nodes(1)))
        print*, 'file ml_weights was loaded'
else
        print*, 'file ml_weights not found'
end if
        
print*, 'durchgelaufen'
        ! Calculate number of bias weights



!############### Initialize Geometry VoF Values ########################
npts(1:3)=10
icphi=1
incst=icphi*nxyza

!circle or sphere
typeOfObject = 'circle'

! Process input variables
tnpts = npts(1)*npts(2)*npts(3)  ! total number of points
nidiv = npts(1)-1                ! number of divisions in i
njdiv = npts(2)-1                ! number of divisions in j
nkdiv = npts(3)-1                ! number of divisions in k
tndiv = nidiv*njdiv*nkdiv        ! total number of divisions

! Allocate and initialize arrays
allocate(gridpoints(tnpts,3))

! Circle/Sphere parameter
radius   = 2d0!0.2d0!
radius1  = 0.01762d0
radius2  = 0.0275d0
center_x = 4d0!0.5d0!
center_y = 4d0!0.5d0!4d0
center_z = 0.0

do m=1,nblo
  call setind(1, m, &
#             include "listsetind.h"
              ier)
  do k=2,nkm
    do i=2,nim
      do j=2,njm
        inp=ha(i,j,k)
        ! Get vertices of the current control volume
        vertices(1,1) = x(ha(i-1,j-1,k-1))
        vertices(1,2) = y(ha(i-1,j-1,k-1))
        vertices(1,3) = z(ha(i-1,j-1,k-1))
        vertices(2,1) = x(ha(i  ,j-1,k-1))
        vertices(2,2) = y(ha(i  ,j-1,k-1))
        vertices(2,3) = z(ha(i  ,j-1,k-1))
        vertices(3,1) = x(ha(i-1,j  ,k-1))
        vertices(3,2) = y(ha(i-1,j  ,k-1))
        vertices(3,3) = z(ha(i-1,j  ,k-1))
        vertices(4,1) = x(ha(i  ,j  ,k-1))
        vertices(4,2) = y(ha(i  ,j  ,k-1))
        vertices(4,3) = z(ha(i  ,j  ,k-1))
        vertices(5,1) = x(ha(i-1,j-1,k  ))
        vertices(5,2) = y(ha(i-1,j-1,k  ))
        vertices(5,3) = z(ha(i-1,j-1,k  ))
        vertices(6,1) = x(ha(i  ,j-1,k  ))
        vertices(6,2) = y(ha(i  ,j-1,k  ))
        vertices(6,3) = z(ha(i  ,j-1,k  ))
        vertices(7,1) = x(ha(i-1,j  ,k  ))
        vertices(7,2) = y(ha(i-1,j  ,k  ))
        vertices(7,3) = z(ha(i-1,j  ,k  ))
        vertices(8,1) = x(ha(i  ,j  ,k  ))
        vertices(8,2) = y(ha(i  ,j  ,k  ))
        vertices(8,3) = z(ha(i  ,j  ,k  ))

        ! Create points distributed evenly in the current control volume
        ipts = 1
        do kdiv=0,nkdiv
          do jdiv=0,njdiv
            do idiv=0,nidiv
              a000 = real((nidiv-idiv) * (njdiv-jdiv) * (nkdiv-kdiv),8) / real(tndiv,8)
              ax00 = real((      idiv) * (njdiv-jdiv) * (nkdiv-kdiv),8) / real(tndiv,8)
              a0y0 = real((nidiv-idiv) * (      jdiv) * (nkdiv-kdiv),8) / real(tndiv,8)
              axy0 = real((      idiv) * (      jdiv) * (nkdiv-kdiv),8) / real(tndiv,8)
              a00z = real((nidiv-idiv) * (njdiv-jdiv) * (      kdiv),8) / real(tndiv,8)
              ax0z = real((      idiv) * (njdiv-jdiv) * (      kdiv),8) / real(tndiv,8)
              a0yz = real((nidiv-idiv) * (      jdiv) * (      kdiv),8) / real(tndiv,8)
              axyz = real((      idiv) * (      jdiv) * (      kdiv),8) / real(tndiv,8)

              gridpoints(ipts,1) = a000*vertices(1,1) + ax00*vertices(2,1) &
                                 + a0y0*vertices(3,1) + axy0*vertices(4,1) &
                                 + a00z*vertices(5,1) + ax0z*vertices(6,1) &
                                 + a0yz*vertices(7,1) + axyz*vertices(8,1)
              gridpoints(ipts,2) = a000*vertices(1,2) + ax00*vertices(2,2) &
                                 + a0y0*vertices(3,2) + axy0*vertices(4,2) &
                                 + a00z*vertices(5,2) + ax0z*vertices(6,2) &
                                 + a0yz*vertices(7,2) + axyz*vertices(8,2)
              gridpoints(ipts,3) = a000*vertices(1,3) + ax00*vertices(2,3) &
                                 + a0y0*vertices(3,3) + axy0*vertices(4,3) &
                                 + a00z*vertices(5,3) + ax0z*vertices(6,3) &
                                 + a0yz*vertices(7,3) + axyz*vertices(8,3)

              ipts = ipts + 1
            end do
          end do
        end do

        ! Check how many points are inside the circle/sphere
        nptsInside = 0
        select case(typeOfObject)
          case('circle')
            do ipts=1,tnpts
              if (((gridpoints(ipts,1)-center_x)**2+(gridpoints(ipts,2)-center_y)**2) < radius**2) then
                nptsInside = nptsInside+1
              end if
            end do
          case('ellipse')
            do ipts=1,tnpts
              if ((((gridpoints(ipts,1)-center_x)/radius1)**2+((gridpoints(ipts,2)-center_y)/radius2)**2 ) <= 1.d0) then
                nptsInside = nptsInside+1
              end if
            end do
          case('sphere')
            do ipts=1,tnpts
              if (((gridpoints(ipts,1)-center_x)**2 &
                  +(gridpoints(ipts,2)-center_y)**2 &
                  +(gridpoints(ipts,3)-center_z)**2) < radius**2) then
                nptsInside = nptsInside+1
              end if
            end do
        end select

        ! Calculate volume fraction
        c(inp) = (real(nptsInside,8) / real(tnpts,8))
        cold(inp)=c(inp)
        c(inp+incst)=1d0-c(inp)
        cold(inp+incst)=1d0-c(inp)

        !p(inp)=dskappa/radius

        !cm(inp)=c(inp)
        !cm(inp+incst)=c(inp+incst)

      end do
    end do
  end do
end do

deallocate(gridpoints)

end subroutine usrini


!#####################################################################
subroutine usrgri( &
#                 include "listall.h" 
                  ier)          
!#####################################################################
! User routine called before changing to a different grid (level).
!#####################################################################
implicit none
#include "cb3dall.h"
#include "cbglobaldim.h"

! dummy variables
#include "dimall.h"
integer :: ier

! local variables
#include "hahihjhk.h"
!#####################################################################

return
end


!#####################################################################
subroutine usrtim( &
#                 include "listall.h" 
                  ier)          
!#####################################################################
! User routine called before each time step.
!#####################################################################
implicit none
#include "cb3dall.h"
#include "cbglobaldim.h"

! dummy variables
#include "dimall.h"
integer :: ier

! local variables
#include "hahihjhk.h"
!#####################################################################

return
end


!#####################################################################
subroutine util(adapdt, &
#               include "listall.h"
                ier)           
!#####################################################################
! User routine called after each time step. For steady calculations called at the end of each grid level (FMG).
!#####################################################################
use adaptime_mod
use line_out
implicit none
#include "cb3dall.h"
#include "cbglobaldim.h"

! Variables of the adaptative time step
type(adaptime_typ), intent(in out) :: adapdt
! dummy variables
#include "dimall.h"
integer :: ier,ngr

! local variables
integer :: i,j,k,m,inp
real*8  :: umax,utemp,utempsum,utempsum2,l1_vel,l2_vel
real*8  :: radius,kappaexact,curvsum,curvmaxtemp,curvmaxtempsum,curvmaxtempsum2
real*8  :: curvmax,curvaverage,l1_curv,l2_curv,curv_error,curv_oszi,curv_oszi_max,curvmax_abs
real*8  :: pressureexact,pinsum,poutsum,pmax,pmin,err_total_p,err_max_p,pinaverage
integer :: incount,outcount,count,countcurv
real*8  :: center_x,center_y,vecx,vecy,normalized,xm,ym,zm,gh,h,linf_curv
real*8  :: lmax_x,lmax_y,capillarnumber,normalnorm2x,normalnorm2y,lnorm2x,lnorm2y
integer :: vecmethod,icphi
logical :: laux
real*8  :: fi1(2*nxyza),fi2(2*nxyza),fi3(2*nxyza),fi4(2*nxyza)
character(len=20) :: filename
real*8  ::  coord_x1,coord_y1,coord_z1,coord_x2,coord_y2,coord_z2
#include "hahihjhk.h"
!#####################################################################
if (it == 1 .and. iproc==1) then
open(69,file="summary.txt",position='append',status='replace')
write(69,456) 'it','innerit','res u','res v','res w','res p','res conc','time timestep','time all'
close(69)
456   FORMAT(12(A, 8X))
end if      

if(iproc==1) then
open(69,file="summary.txt",position='append')
write(69,457) it,ls,resor(iu),resor(iv),resor(iw),resor(ip),resor(icon),deltim,tetime
close(69)
457   FORMAT(i4,2x,i4,2x,e11.4,2x,e11.4,2x,e11.4,2x,e11.4,2x,e11.4,2x,f15.5,2x,f15.5)
end if

! Define the errors of umax
umax=0d0
utemp=0d0
utempsum=0d0
utempsum2=0d0
count=0
l1_vel=0d0
l2_vel=0d0

do m=1,nblo
  call setind(1,m, &
#             include "listsetind.h"            
              ier)            
  do k=2,nkm
    do i=2,nim
      do j=2,njm
        inp=ha(i,j,k)
        utemp=abs(sqrt(u(inp)**2+v(inp)**2))
        utempsum=utempsum+utemp
        utempsum2=utempsum2+utemp**2
        if ( utemp > umax) then
          umax=utemp
        end if
        count=count+1
      end do
    end do
  end do
end do 

capillarnumber=umax*0.1d0
print *,capillarnumber

! error norms
! best =0
l1_vel=utempsum/dble(count)
! best =0
l2_vel=sqrt(utempsum2)/sqrt(dble(count))

! Define the errors of curvature
radius=2d0
kappaexact=1d0/radius
curvsum=0d0
curvmaxtemp=0d0
curvmaxtempsum=0d0
curvmaxtempsum2=0d0
curvmax=0d0
curvmax_abs=0d0
curv_oszi_max=0d0
curvaverage=0d0
l1_curv=0d0
l2_curv=0d0
curv_error=0d0
curv_oszi=0d0
countcurv=0
do m=1,nblo
    call setind(1,m, &
#               include "listsetind.h"            
                ier)            
  do k=2,nkm
    do i=2,nim
      do j=2,njm
        inp=ha(i,j,k)
        if (dscurv2(inp)/= 0d0) then
          curvsum=curvsum+dscurv2(inp)
          curvmaxtemp=abs(abs(dscurv2(inp))-kappaexact)
          curvmaxtempsum=curvmaxtempsum+curvmaxtemp
          curvmaxtempsum2=curvmaxtempsum2+(curvmaxtemp/kappaexact)**2
          if (abs(curvmaxtemp) > abs(curvmax)) then
            curvmax=curvmaxtemp
          end if
          if (abs(dscurv2(inp))>abs(curvmax_abs)) then
            curvmax_abs=abs(dscurv2(inp))
          end if
          countcurv=countcurv+1
        end if
      end do
    end do
  end do
end do 


! error norms
! best =kappaexact
curvaverage=curvsum/dble(countcurv)
! best =0 
curv_error=abs(curvaverage-kappaexact)/kappaexact
! best =1
curv_oszi=abs(abs(curvaverage)-abs(curvmax))/abs(curvaverage)
! best =0
curv_oszi_max=abs(curvmax_abs-kappaexact)/kappaexact
! best =0
l1_curv=curvmaxtempsum/dble(countcurv)
! best =0
l2_curv=sqrt(curvmaxtempsum2/(dble(countcurv)))
! best = 0
linf_curv=curvmax/kappaexact

! Define the errors of pressure
pressureexact=dskappa/radius
pinsum=0d0
poutsum=0d0
pmax=0d0
pmin=1e10
incount=0
outcount=0
err_total_p=0d0
err_max_p=0d0
pinaverage=0d0
do m=1,nblo
    call setind(1,m, &
#               include "listsetind.h"            
                ier)            
  do k=2,nkm
    do i=2,nim
      do j=2,njm
        inp=ha(i,j,k)
        if (c(inp)>0.99d0) then
          pinsum=pinsum+p(inp)
          incount=incount+1
        else if (c(inp)<0.01d0) then
          poutsum=poutsum+p(inp)
          outcount=outcount+1
        end if
        if (p(inp) > pmax) then
          pmax=p(inp)
        else if (p(inp)<pmin) then
          pmin=p(inp)
        end if
      end do
    end do
  end do
end do 


! error norms
pinaverage=pinsum/dble(incount)
err_total_p=(abs((pinsum/dble(incount))-(poutsum/dble(outcount)))-pressureexact)/pressureexact
err_max_p=(abs(pmax-pmin)-pressureexact)/pressureexact

if (nproc > 1) then
  call maxbro (umax,ier)
  call maxbro (l1_vel,ier)
  call maxbro (l2_vel,ier)
  call sumbro (curvaverage,ier)
  curvaverage=curvaverage/nproc
  call maxbro (curv_error,ier)
  call minbro (curv_oszi,ier)
  call maxbro (curv_oszi_max,ier)
  call maxbro (l1_curv,ier)
  call maxbro (l2_curv,ier)
  call maxbro (pinaverage,ier)
  call maxbro (abs(err_total_p),ier)
  call maxbro (abs(err_max_p),ier)
end if

print *, 'Average CURVATUREE', curvaverage

if (it == 1 .and. iproc==1) then
open(69,file="staticbubble_results.txt",position='append',status='replace')
write(69,459) 'it','umax','l1_umax','l2_umax','linf_curv','curvaverage','average_error','curv_oszi','curv_oszi_max',&
              'l1_curv','l2_curv','press_in','press_exact','total_err','max_err'
close(69)
459   FORMAT(15(A, 8X))
end if      

if(iproc==1) then
open(69,file="staticbubble_results.txt",position='append')
write(69,460) it,umax,l1_vel,l2_vel,linf_curv,curvaverage,curv_error,curv_oszi,curv_oszi_max,l1_curv,l2_curv,&
              pinaverage,pressureexact,err_total_p,err_max_p
close(69)
460   FORMAT(i4,2x,14(e13.6,4x))
end if

if (it == 1 .and. iproc==1) then
open(69,file="staticbubble_results_reduced.txt",position='append',status='replace')
write(69,459) 'it','umax','linf_curv','l2_curv','error_pminmax','error_pinout'
close(69)
end if      

if(iproc==1) then
open(69,file="staticbubble_results_reduced.txt",position='append')
write(69,460) it,umax,linf_curv,l2_curv,err_max_p,err_total_p
close(69)
end if

if (iproc==1) then
  open(1234, file='conc_curv_forplot.txt',position='append',status='replace')
  if (it==1) then
    write(1234,*) 'xc   ','yc   ','conc   ','kappa   '
  end if
  do m=1,nblo
      call setind(1,m, &
#               include "listsetind.h"            
                ier)            
      do k=2,nkm
        do i=1,ni
          do j=1,nj
            inp=ha(i,j,k)
            write(1234,1235) xc(inp),yc(inp),c(inp),dscurv2(inp)
          end do
        end do
      end do
    end do            
  close(1234)
end if
1235 FORMAT(4(f7.4,4x))

! Compare normals
do icphi=0,mcphi
    call chebyshev2(1,icphi,2,3,1,c,fi1,fi2,fi3,fi4,.false.,.false., &
#                  include "listcalcurv.h"
                   ier)
end do
vecmethod=6
!call calgradc(1,0,vecmethod,fi4,dscdx,dscdy,dscdz, &
!#             include "listcalgradc.h"
!              ier)
call calgrad_lsf(1,0,fi4,dscdx,dscdy,dscdz,dscdxnorm,dscdynorm,dscdznorm, &
#                include "listcalgradc.h"
                 ier)


center_x=4d0
center_y=4d0
lmax_x=0d0
lmax_y=0d0
count=0
lnorm2x=0d0
lnorm2x=0d0
do m=1,nblo
    call setind(1,m, &
#               include "listsetind.h"            
                ier)            
  do k=2,nkm
    do i=2,nim
      do j=2,njm
        inp=ha(i,j,k)
        if (c(inp)<0.99d0 .and. c(inp)>0.01d0) then
          count=count+1
          !Vector from center to points
          vecx=(center_x-xc(inp))
          vecy=(center_y-yc(inp))
          normalized=sqrt(vecx**2+vecy**2)
          realnormalx(inp)=vecx/normalized
          realnormaly(inp)=vecy/normalized
          normalized=sqrt(dscdx(inp)**2+dscdy(inp)**2)
          newnormalx(inp)=dscdx(inp)/normalized
          newnormaly(inp)=dscdy(inp)/normalized
          lmax_xtemp(inp)=abs(realnormalx(inp)-newnormalx(inp))
          lmax_ytemp(inp)=abs(realnormaly(inp)-newnormaly(inp))
          !print *,xc(inp),yc(inp),lmax_xtemp(inp)
          lnorm2x=lnorm2x+lmax_xtemp(inp)**2
          lnorm2y=lnorm2y+lmax_ytemp(inp)**2
          !if (c(inp)<0.99d0 .and. c(inp)>0.01d0) then
            if (lmax_xtemp(inp)>lmax_x) then
              lmax_x=lmax_xtemp(inp)
            endif
            if (lmax_ytemp(inp)>lmax_y) then
              lmax_y=lmax_ytemp(inp)
            endif
          !endif
          !if (c(inp)<0.99d0 .and. c(inp)>0.01d0) then
             !print *,c(inp),cm(inp),vecx,dscdx(inp),vecy,dscdy(inp)
           !  print *,c(inp),cm(inp),realnormalx(inp),newnormalx(inp),realnormaly(inp),newnormaly(inp)
          !end if
          !if (lmax_x > 1.9999d0) then
          !  STOP
          !end if
        end if
      end do
    end do
  end do
end do
print *,count,lmax_x,lmax_y
normalnorm2x=sqrt(1/real(count)*lnorm2x)
normalnorm2y=sqrt(1/real(count)*lnorm2y)

if (it == 1 .and. iproc==1) then
open(69,file="normalvec_res.txt",position='append',status='replace')
write(69,501) 'method','grid','lmax_x','lmax_y','L2_normx','L2_normy'
close(69)
501   FORMAT(6(A, 8X))
end if      

if(iproc==1) then
open(69,file="normalvec_res.txt",position='append')
write(69,502) vecmethod,ni-2,lmax_x,lmax_y,normalnorm2x,normalnorm2y
close(69)
502   FORMAT(i4,6x,i4,2x,4(e16.8,4x))
end if

itline=itline+1

l_flag(1)='x'
l_flag(2)='y'
l_flag(3)='c'
l_flag(4)='p'
l_flag(5)='u'
l_flag(6)='v'

filename='middleline'
coord_x1=0.1d0
coord_y1=3.9d0
coord_z1=0d0
coord_x2=7.9d0
coord_y2=3.9d0
coord_z2=0d0
call writeoutput((/coord_x1,coord_y1,coord_z1/),(/coord_x2,coord_y2,coord_z2/),ni,0,filename,&
#                include "listall.h"
                 ier)

filename='diagonalline'
coord_x1=0.1d0
coord_y1=0.1d0
coord_z1=0d0
coord_x2=7.9d0
coord_y2=7.9d0
coord_z2=0d0
call writeoutput((/coord_x1,coord_y1,coord_z1/),(/coord_x2,coord_y2,coord_z2/),2*ni,0,filename,&
#                include "listall.h"
                 ier)

return
end

!#####################################################################
subroutine usritr( &
#                 include "listall.h" 
                  ier)          
!#####################################################################
! User routine called before every V-cycle.
!#####################################################################
implicit none
#include "cb3dall.h"
#include "cbglobaldim.h"

! dummy variables
#include "dimall.h"
integer :: ier

! local variables
#include "hahihjhk.h"
!#####################################################################

return
end


!#####################################################################
subroutine usrlsg( &
#                 include "listall.h"  
                  ier)         
!#####################################################################
! User routine called before every fine grid iteration of a V-cycle.
!#####################################################################
implicit none
#include "cb3dall.h"
#include "cbglobaldim.h"

! dummy variables
#include "dimall.h"
integer :: ier

! local variables
#include "hahihjhk.h"
!#####################################################################

return
end
