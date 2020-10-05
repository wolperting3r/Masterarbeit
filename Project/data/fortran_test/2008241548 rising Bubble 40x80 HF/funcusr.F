#include "listdefines.h"
!#####################################################################
subroutine usrini(ngr, adapdt, &
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

! dummy variables
type(adaptime_typ), intent(in) :: adapdt ! 
integer, intent(in) :: ngr
#include "dimall.h"
integer :: ier

! local variables
integer :: i,j,k,m,inp,ittemp,itimes
integer :: icphi,incst,incst1
real*8 :: fden,fden2,fvis,fvis2

! local variables for VOF distribution
real*8 :: xr,yr,zr,rb,rc,franzx(8),franzcount,phi(4),phic,FranzA,FranzB1,FranzB2,FranzA2
real*8 :: schenk(6),Farea,ellipse1,ellipse2, y_water
real*8 :: r(4),r1
integer :: temp1

! arguments
integer                             :: npts(3)

! ML
integer :: stat  ! status of file io
integer                                 :: ml_totalweightlength
integer                                 :: ml_totalbiaslength
integer                                 :: ml_linecounter_layer
integer                                 :: ml_linecounter_bias
integer :: stz


! local variables
real*8, dimension(:,:), allocatable :: gridpoints
real*8                              :: vertices(8,3)
real*8                              :: radius,center_x,center_y,center_z
real*8                              :: a000,ax00,a0y0,axy0,a00z,ax0z,a0yz,axyz
integer                             :: idiv,jdiv,kdiv,ipts
integer                             :: nidiv,njdiv,nkdiv
integer                             :: tnpts,tndiv,nptsInside
character(len=6)                    :: typeOfObject
logical                             :: activ
#include "hahihjhk.h"
!#######################################################################
loop_counter = 1
open(unit=142,file=&
  'model_mlp_1000_200-150-120_9x9_rot_ellipse_flp_cut_dshift1_shift1_bia_int1_edg_ncl_s1_100_CVOFLS',&
  action='read',status='old',iostat=stat)
if (stat == 0) then
       allocate(ml_tmp_curv(40, 80))
       allocate(ml_last_curv(40, 80))
       ! initialize ml_tmp_curv
       do m=1,nblo
       call setind(ngr,m, &
#                   include "listsetind.h"
                   ier)
         do k=2,nkm
           do i=2,nim
             do j=2,njm
               ml_tmp_curv(i-1, j-1) = 0
               ml_last_curv(i-1, j-1) = 0
             end do
           end do
         end do
       end do


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
       ! calculate total length of node weights
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
if (lconc) then
  call setzero(c,nxyzac,1,nxyzac)
  call setzero(cold,nxyzac,1,nxyzac)
  activ=.true.
end if

if (.not. activ) then
  icphi=1
  incst=icphi*nxyza
  do m=1,nblo
    call setind(1, m, &
#               include "listsetind.h"
                ier)
    do k=2,nkm
      do i=2,nim
        do j=2,njm
          inp=ha(i,j,k)

          ! Calculate volume fraction
          c(inp) = 1.d0
          c(inp+incst)=0.d0
          p(inp)=0.d0
        end do
      end do
    end do
  end do

else

  npts(1:3)=10
  icphi=1
  incst=icphi*nxyza

  !circle or sphere or plane
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
  radius   = 0.25
  center_x=0.5d0
  center_y=0.5d0

  !center_z = 0.0115
  !y_water=3.0d0

  do m=1,nblo
    call setind(1, m, &
#               include "listsetind.h"
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

          ! Check how many points are inside the circle/sphere/plane
          nptsInside = 0
          select case(typeOfObject)
            case('circle')
              do ipts=1,tnpts
                if (((gridpoints(ipts,1)-center_x)**2+(gridpoints(ipts,2)-center_y)**2) < radius**2) then
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
            case('plane')
              do ipts=1,tnpts
                if (gridpoints(ipts,2) < y_water) then
                  nptsInside = nptsInside+1
                end if
              end do
          end select

          ! Calculate volume fraction
          c(inp) = (real(nptsInside,8) / real(tnpts,8))
          c(inp+incst)=1.d0-c(inp)
        end do
      end do
    end do
  end do

  deallocate(gridpoints)

end if

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

end


!#####################################################################
subroutine util(adapdt, &
#               include "listall.h"
                ier)           
!#####################################################################
! User routine called after each time step. For steady calculations called at the end of each grid level (FMG).
!#####################################################################
use adaptime_mod
implicit none
#include "cb3dall.h"
#include "cbglobaldim.h"

! dummy variables
! Variables of the adaptative time step
type(adaptime_typ), intent(in) :: adapdt
#include "dimall.h"
integer :: ier

! local variables
real*8 :: mass,xc_mass,yc_mass,volumen,perimetro,sumvel,den_aire
real*8 :: x_center,y_center,vrise,mass_local,vol_local
integer :: ivec, inp,ine,inw,inn,ins,m,i,j
! local parameters
real*8, parameter :: pi=3.1415926535897932384626433832795028841971693d0
! local variables
logical :: ltimreport
#include "hahihjhk.h"
!#####################################################################

! ltimreport=.true.
! if (ltimreport) then
  ! if (ls==1) ls=ls
  ! if (ls>1)  ls=ls-1
  ! if (iproc==1) then
    ! if (it==1 .or. .not.ltime) then
      ! OPEN(unit=1022,file='out/hist_error.dat',action='write',status='replace')
        ! write(1022,610)
        ! write(1022,2000) it,dtime,totime,adapdt%cflmax,ls,iterf,itc,itct,resor(iu),resor(iv),resor(iw),resor(ip),resor(icon),&
                         ! deltim,tetime
      ! close(1022)
    ! else
      ! OPEN(unit=1022,file='out/hist_error.dat',action='write',position='append')
        ! write(1022,2000) it,dtime,totime,adapdt%cflmax,ls,iterf,itc,itct,resor(iu),resor(iv),resor(iw),resor(ip),resor(icon),&
                         ! deltim,tetime
     ! close(1022)
   ! end if
  ! end if
! end if

! 2000 format(i6,2x,1p,3e10.2,1x,i7,1x,i15,1x,i15,1x,i15,2x,1p,7e12.2)
! 610 format(4x,'it',6x,'dtime',4x,'totime',4x,'maxcfl',7x,'ls',6x,'Total_iter',15x,'itc',11x,'itct',5x, &
           ! 'umom',8x,'vmom',8x,'wmom',8x,'mass',8x,'conc',8x,'time',8x,'total_time')

!center of the mass and mean velocity
mass=0.d0
xc_mass=0.d0
yc_mass=0.d0
volumen=0.d0
sumvel=0.d0
den_aire=100.d0

do m=1,nblo
  call setind(1,m, &
#             include "listsetind.h"
              ier)   
  do ivec = iis,iie
    inp=ijkvec(ivec)
    if (c(inp)>=0.5d0) then
      mass_local = c(inp)*vol(inp)*den_aire
      vol_local = c(inp)*vol(inp)
      xc_mass = xc_mass + vol_local*xc(inp)
      yc_mass = yc_mass + vol_local*yc(inp)
      sumvel = sumvel + vol_local*v(inp)
      mass = mass + mass_local
      volumen = volumen + c(inp)*vol(inp)
    end if
  end do
end do

call sumbro(xc_mass,ier)
call sumbro(yc_mass,ier)
call sumbro(mass,ier)
call sumbro(volumen,ier)

call sumbro(sumvel,ier)

x_center=xc_mass/volumen
y_center=yc_mass/volumen
vrise=sumvel/volumen

call brcastr8(1,x_center,1, ier)
call brcastr8(1,y_center,1, ier)
call brcastr8(1,vrise,1, ier)

if (iproc==1) then
  if (it==1) then 
    OPEN(unit=1023,file='out/parametros.dat',action='write',status='replace')
    write(1023,630)
    close(1023)
  end if

  OPEN(unit=1023,file='out/parametros.dat',action='write',position='append')
  write(1023,2002) it,totime,mass,y_center,vrise
  close(1023)
end if

2002 format(i6,2x,1p,6e12.4)
630 format(4x,'it',4x,'totime',6x,'mass',8x,'y_center',4x,'vel_rise')

! values for circularity
if (it == 1 .and. iproc==1) then
  open(1024,file="out/xy_position.dat",status='replace')
    write(1024,461) 'time','xpos','ypos','proc'
  close(1024)
end if

461   FORMAT(4(A, 17X))

do i=1,1 
if (iproc==i) then
open(1024,file="out/xy_position.dat",position='append')
do m=1,nblo
  call setind(1,m, &
#             include "listsetind.h"
              ier)   
  do ivec = iis,iie
    inp=ijkvec(ivec)
    if (mod(it,160)==0 .or. it == 1) then
      if (c(inp)>0.1d0 .and. c(inp)<0.9d0) then
        write(1024,460) totime, xc(inp),yc(inp),iproc
      end if
    end if
  end do
end do
close(1024)
end if
end do

460 FORMAT(3f17.12,i6)

end subroutine util

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
