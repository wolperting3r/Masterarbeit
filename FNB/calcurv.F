#include "listdefines.h"
!####################################################################################################################################
subroutine calcurv(ngr, &
#                  include "listcalcurv.h"
                   ier)
!####################################################################################################################################
! Wrapper for volume fraction curvature computation
!####################################################################################################################################
implicit none
#include "cb3dall.h"
#include "cbglobaldim.h"

! dummy variables
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

! dummy variables
integer,intent(in) :: ngr
integer:: ier
      
! local variables
logical :: lcon,lylr,lflr,lylrf,lmix,lfy,lfend
integer :: icphi,incst,inp,m,i,j,k,numa,numb,numc,numa1,numb1,numc1,count1,incst1
real*8  :: ch1,ch2,ch3,ch4
real*8  :: fi1(nxyza),fi2(nxyza),fi3(nxyza),fi4(nxyza)
real*8  :: dscurv2a, dscurvsum2

#include "hahihjhk.h"
!####################################################################################################################################
! Standard methods:
  ! 1 : CST
  ! 2 : CST-Hybrid (1.CST+2.CDS)
  ! 3 : No surface tension
  ! 4 : CDS
  ! 5 : LocalRadius
  ! 6 : CST+CDS (1.CST+2.CST + CDS at borders)
  ! 7 : CVOFLS
  ! 8 : Young
  ! 9 : Young+Filter
  !10 : HeightFunction
  !11 : Fixed curvature (only for bubble/drop) Use as a reference to check the accuracy of curvature models
!####################################################################################################################################
! Moved from prepstep (if steady)
if (.not.ltime) then
  if (lconc) then
    do icphi=0,mcphi
      incst=icphi*nxyza
      do inp=incst+icst,incst+icen
        cold(inp)=c(inp)
        dsdirac(inp)=0.d0
      end do
    end do
  end if
end if

! Extrapolation to boundaries
do icphi=0,mcphi
  incst1=(icphi*nxyza)+1
  call bconc(c(incst1),ngr, &
#          include "listbpres.h"
           ier)
end do

do m=1,nblo
  call setind(ngr,m, &
#             include "listsetind.h"
              ier)
  do k=2,nkm
    do j=2,njm
      do i=2,nim
        inp=ha(i,j,k)
        dsdirac(inp)=1.d0
      end do
    end do
  end do
end do

! Set zero "old" curvature values - necessary because when interface moves
! condition for curvature is computed only in interface region
call setzero(dscurv2,nxyza,1,nxyza,ier)
! Check if thirddim was read right
if (thirddim == 3) then
  STOP 'thirddim is wrong!'
end if
!#################################################################################################################################### 
select case (lst)
  case (1)
    !Curvature method: CST
    call curv_cst(ngr, & 
#                 include "listcalcurv.h"
                  ier)

  case (2)
    !Curvature method: CST-Hybrid
    call curv_cstkwp(ngr,icphi, & 
#                    include "listcalcurv.h"
                     ier)

  case (3) 
    !No Surface Tension calculation
    dskappa=0.d0

  case (4)
    !Curvature method: CDS - Standard Finite-Difference-Method
    lcon=.true.
    lmix=.false.
    call curv_cds(ngr,lcon,lmix,dscurv2a, &
#                 include "listcalcurv.h"
                  ier)

  case (5)
    !Curvature method: Local Radius method
    lylr=.false.
    lflr=.false.
    lylrf=.false.
    flcell2=0
    call curv_lr(ngr,icphi,lylr,lflr,lylrf,cm,dscurv2, &
#                include "listcalcurv.h"
                 ier)

  case (6)
    !Curvature method: CST+CDS
    call curv_cstcds(ngr,icphi, &
#                    include "listcalcurv.h"
                     ier)

  case (7)
    !Curvature method: CVOFLS
    call curv_ls(ngr, &
#                include "listcalcurv.h"
                 ier)
    
  case (8)
    !Curvature method: Young
    numa=1
    numb=3
    lylr=.false.
    lmix=.false.
    lfy=.false.

    call curv_young(ngr,lylr,lmix,lfy,numa,numb,c, &
#                   include "listcalcurv.h"
                    ier)


  case (9)
    !Curvature method: Young+Filter
    do m=1,nblo
      call setind(ngr,m, &
#                 include "listsetind.h"
                  ier)
      do k=1,nk
        do i=1,ni
          do j=1,nj
            inp=ha(i,j,k)
            fi1(inp)=c(inp)
            fi2(inp)=c(inp)
            fi3(inp)=c(inp)
            fi4(inp)=c(inp)
          end do
        end do
      end do
    end do ! m

  
    numa=4
    numb=2
    numa1=1
    numb1=3

    lfy=.true. 
    lfend=.false.
    ! 1-lsp0,2-binominal,3-5pointcheby,4-lsp2
    call chebyshev(ngr,1,numa,numb,c,fi1,fi2,fi3,fi4,lfy,lfend, &
#                  include "listcalcurv.h"
                   ier)
    lylr=.false.
    lmix=.false.
    call curv_young (ngr,lylr,lmix,lfy,numa1,numb1,fi4, &
#                      include "listcalcurv.h"
                       ier)

  case (10)
    !Curvature method: Height-Functions
    if (thirddim==1) then
      call curv_hf(ngr, &
#                  include "listcalcurv.h"
                   ier)
    else
      call curv_hf2D(ngr, &
#                    include "listcalcurv.h"
                     ier)
    end if

  case (11)
    !Fixed curvature (only for bubble/drop) Use to mesure the accuracy of the methods
    call curv_cdshc(ngr,icphi, &
#                   include "listcalcurv.h"
                    ier)
  case (12)
    ! Derivatives with least square fits
    call curv_celeste(ngr,dscurv2a, &
#                   include "listcalcurv.h"
                    ier)
  case (13)
    ! Machine Learning
    call curv_ml(ngr, &
#                   include "listcalcurv.h"
                    ier)

end select
end subroutine calcurv

!####################################################################################################################################
subroutine curvature (ngr,lcon,lmix,lfy,dscurv2a, &
#                     include "listcalcurv.h"
                      ier)
!####################################################################################################################################
! Calculation of the curvature
!####################################################################################################################################
implicit none
#include "cb3dall.h"
#include "cbglobaldim.h"

! dummy variables
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

integer :: ier,icphi

! dummy variables
logical, intent(in) ::lcon,lmix,lfy
integer,intent(in)  :: ngr
real*8, intent (out):: dscurv2a
      
! local variables
logical :: laux,lfend,curvfiltering,curvweightening,secondweighting,flatsurfacectrl,nolocalcurv
integer :: k1,kn,i1,inn,j1,jn,kstart,kend,numa,numb,incst,incst1
integer :: i,j,k,l,m,count1,inp,checkit,zerocount
real*8  :: vz,xm,ym,zm,h,gh,dscurvsum2
integer :: curvfilter,weightsexp,weightsexp2
real*8  :: tempcurv(nxyza),tempcurv2(nxyza),weights(nxyza),normalized
real*8  :: fi1(nxyza),fi2(nxyza),fi3(nxyza),fi4(nxyza)
real*8  :: xclim1,xclim2,xclim1_3d,xclim2_3d,scalingfactor
#include "hahihjhk.h"
!####################################################################################################################################
scalingfactor=0.5d0

! Options for curvature manipulation
flatsurfacectrl=.false.
curvfiltering=.false.
 !1-lsp0,2-binominal,3-5pointcheby,4-lsp2
curvfilter=3
curvweightening=.false.
weightsexp=1
secondweighting=.false.
weightsexp2=1
! no local curvature, eq at boundaries
nolocalcurv=.false.
! values in case of dropimpact_bias320 - those are the last two cells before the border
!xclim1_3d=0.00275d0
!xclim2_3d=0.02951d0
xclim1_3d=(0.00065d0)*scalingfactor
xclim2_3d=((0.01d0-0.00065d0))*scalingfactor
! values for surface osci 128
xclim1=(0.00065d0)*scalingfactor
xclim2=((0.01d0-0.00065d0))*scalingfactor

vz=-1.d0
dscurvsum2=0.d0
count1=0

do m=1,nblo
  call setind(ngr,m, &
#             include "listsetind.h"
              ier)

  i1=2
  inn=nim
  j1=2
  jn=njm
  k1=2
  
  if (thirddim == 1) then
    kn=nkm
    if (lfy) then
      i1=3
      inn=nim-1
      j1=3
      jn=njm-1
      k1=3
      kn=nkm-1
   end if
  else if (thirddim == 0) then
    kn=2
    if (lfy) then
      i1=3
      inn=nim-1
      j1=3
      jn=njm-1
   end if
  end if

  do k=k1,kn
    do j=j1,jn
      do i=i1,inn
        inp=ha(i,j,k)
        xm=sqrt((x(inp)-x(inp-nj))**2)
        ym=sqrt((y(inp)-y(inp-1))**2)
        zm=sqrt((z(inp)-z(inp-nij))**2)

        if (thirddim == 1) then 
          h=min(xm,ym,zm)
        else if (thirddim == 0) then
          h=min(xm,ym)
        end if

        gh=0.01d0/h
        
        laux=.false.

        if (thirddim == 1) then
          if (sqrt(dscdx(inp)**2+dscdy(inp)**2+dscdz(inp)**2) > gh) then
            if ((lcon).and.(cm(inp) > 0.01d0).and.(cm(inp) < 0.99d0)) then
              laux=.true.
            else if (.not.lcon) then
              laux=.true.
            end if
          end if

        else if (thirddim == 0) then
          if ((sqrt(dscdx(inp)**2+dscdy(inp)**2) > gh)) then  
            if ((lcon).and.(cm(inp) > 0.01d0).and.(cm(inp) < 0.99d0)) then 
              laux=.true.
            else if (.not.lcon) then
              laux=.true.
            end if
          end if
        end if

        if (laux) then
          dscurv2(inp)= vz*(dscdy(inp)**2*dscdxx(inp)-2*dscdx(inp)*dscdy(inp)*dscdxy(inp)+dscdx(inp)**2*dscdyy(inp)+ &
                            dscdx(inp)**2*dscdzz(inp)-2*dscdx(inp)*dscdz(inp)*dscdxz(inp)+dscdz(inp)**2*dscdxx(inp)+ &
                            dscdy(inp)**2*dscdzz(inp)-2*dscdy(inp)*dscdz(inp)*dscdyz(inp)+dscdz(inp)**2*dscdyy(inp))/ &
                           (dscdx(inp)**2+dscdy(inp)**2+dscdz(inp)**2)**(3.d0/2.d0)

          ! Without applying the rule of Schwartz
         !dscurv2(inp)=vz*(dscdxx(inp)*dscdy(inp)**2+dscdxx(inp)*dscdz(inp)**2-dscdx(inp)*dscdy(inp)*dscdyx(inp)- &
                          !dscdx(inp)*dscdz(inp)*dscdzx(inp)+dscdyy(inp)*dscdx(inp)**2+dscdyy(inp)*dscdz(inp)**2- &
                          !dscdy(inp)*dscdx(inp)*dscdxy(inp)-dscdy(inp)*dscdz(inp)*dscdzy(inp)+dscdzz(inp)*dscdx(inp)**2 + &
                          !dscdzz(inp)*dscdy(inp)**2-dscdz(inp)*dscdx(inp)*dscdxz(inp)-dscdz(inp)*dscdy(inp)*dscdyz(inp))/ &
                          !(dscdx(inp)**2+dscdy(inp)**2+dscdz(inp)**2)**(3./2.)

          count1=count1+1
          dscurvsum2=dscurvsum2+dscurv2(inp)      
        else 
          dscurv2(inp)=0.d0
        end if
      end do
    end do
  end do
end do

! Control if there might be a flat surface = no curvature
! If two spatial derivatives (and two neighbors) of the concentration are very small, there is a surface
if (flatsurfacectrl .eqv. .true.) then
  checkit=0
  zerocount=0
  do m=1,nblo
      call setind(ngr,m, &
#                include "listsetind.h"            
                  ier)            
    do k=2,nkm
      do i=2,nim
        do j=2,njm
          inp=ha(i,j,k)  
          ! Control if there might be a flat surface = no curvature
          ! If two spatial derivatives of the concentration are very small, there is a surface
          if (c(inp) > 0.01d0 .and. c(inp) < 0.99d0) then
            ! y-plane
            if (abs(dscdx(inp))<1.d0 .and. abs(dscdz(inp))<1.d0) then
              do l=-1,1
                if (abs(dscdx(ha(i+l,j,k)))<1.d0 .and. abs(dscdz(ha(i+l,j,k)))<1.d0) then
                  checkit=checkit+1
                end if
              end do
              if (checkit == 3 ) then
                !print *,'dx/dz',m,i,j,k,inp,c(inp),dscurv2(inp),dscdx(inp),dscdz(inp)
                dscurv2(inp)=0.d0
                zerocount=zerocount+1
              end if
              checkit=0
  
            ! x-plane
            else if (abs(dscdy(inp))<1.d0 .and. abs(dscdz(inp))<1.d0) then
              do l=-1,1
                if (abs(dscdy(ha(i,j+l,k)))<1.d0 .and. abs(dscdz(ha(i,j+l,k)))<1.d0) then
                  checkit=checkit+1
                end if
              end do
              if (checkit == 3 ) then
                !print *,'dy/dz',m,i,j,k,inp,c(inp),dscurv2(inp),dscdy(inp),dscdz(inp)
                dscurv2(inp)=0.d0
                zerocount=zerocount+1
              end if
              checkit=0
            ! z-plane       
            else if (abs(dscdx(inp))<1.d0 .and. abs(dscdy(inp))<1.d0) then
              do l=-1,1
                if (abs(dscdx(ha(i+l,j,k)))<1.d0 .and. abs(dscdy(ha(i+l,j,k)))<1.d0) then
                  checkit=checkit+1
                end if
              end do
              if (checkit == 3 ) then
                !print *,'dx/dy',m,i,j,k,inp,c(inp),dscurv2(inp),dscdx(inp),dscdy(inp)
                dscurv2(inp)=0.d0
                zerocount=zerocount+1
              end if
              checkit=0
            end if
          end if
        end do
      end do
    end do
  end do
end if

! Filter the curvature afterwards
if (curvfiltering .eqv. .true.) then
  call exall(ngr,dscurv2,1,1, &
#            include "listexall.h"
             ier)
  lfend=.false.
  numa=3
  numb=1
  !1-lsp0,2-binominal,3-5pointcheby,4-lsp2
  call chebyshev(ngr,curvfilter,numa,numb,dscurv2,fi1,fi2,fi3,fi4,lfy,lfend, &
#                include "listcalcurv.h"
                 ier)
  do m=1,nblo
    call setind(ngr,m, &
#               include "listsetind.h"
                ier)
    do k=2,nkm
      do i=2,nim
        do j=2,njm
          inp=ha(i,j,k)
          !if (cm(inp)<(1d0-eps) .and. cm(inp)>(eps))then
          !if (cm(inp)>0d0 .and. cm(inp)<1d0)then
          if (dscurv2(inp)/=0d0)then
            dscurv2(inp)=fi4(inp)
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
          weights(inp)=(1-2d0*abs(0.5d0-cm(inp)))**(weightsexp)
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
                dscurv2(ha(i+1,j,k))*weights(ha(i+1,j,k))+ &
                dscurv2(ha(i-1,j,k))*weights(ha(i-1,j,k))+ &
                dscurv2(ha(i,j+1,k))*weights(ha(i,j+1,k))+ &
                dscurv2(ha(i,j-1,k))*weights(ha(i,j-1,k)))/ &
                (weights(inp)+weights(ha(i+1,j,k))+weights(ha(i-1,j,k))+weights(ha(i,j+1,k))+weights(ha(i,j-1,k)))
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
          if (dscurv2(inp)/=0d0) then
            dscurv2(inp)=tempcurv(inp)
          end if
        end do
      end do
    end do
  end do
end if

! Second step
if (secondweighting .eqv. .true.) then
  call setzero(weights,nxyza,1,nxyza,ier)
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
          weights(inp)=(1-2d0*abs(0.5d0-cm(inp)))**(weightsexp)
        end do
      end do
    end do
  end do
  call setzero(tempcurv2,nxyza,1,nxyza,ier)
  normalized=0d0
  do m=1,nblo
    call setind(ngr,m, &
#                 include "listsetind.h"
                  ier)
    do k=2,nkm
      do i=2,nim
        do j=2,njm
          inp=ha(i,j,k)
          if (dscurv2(inp)/=0d0) then
            if (thirddim ==0) then  
              normalized=sqrt(dscdx(inp)**2+dscdy(inp)**2)
              tempcurv2(inp)=dscurv2(inp)*weights(inp) + &
                (dscurv2(ha(i+1,j,k))*weights(ha(i+1,j,k))*abs(dscdx(inp)/normalized)**(weightsexp2) + &
                 dscurv2(ha(i-1,j,k))*weights(ha(i-1,j,k))*abs(dscdx(inp)/normalized)**(weightsexp2) + &
                 dscurv2(ha(i,j+1,k))*weights(ha(i,j+1,k))*abs(dscdy(inp)/normalized)**(weightsexp2) + &
                 dscurv2(ha(i,j-1,k))*weights(ha(i,j-1,k))*abs(dscdy(inp)/normalized)**(weightsexp2) ) / &
                (weights(inp)+ &
                 weights(ha(i+1,j,k))*abs(dscdx(inp)/normalized)**(weightsexp2) + &
                 weights(ha(i-1,j,k))*abs(dscdx(inp)/normalized)**(weightsexp2) + &
                 weights(ha(i,j+1,k))*abs(dscdy(inp)/normalized)**(weightsexp2) +  &
                 weights(ha(i,j-1,k))*abs(dscdy(inp)/normalized)**(weightsexp2))
              !tempcurv2(inp)=dscurv2(inp)*weights(inp) + &
              !  (dscurv2(ha(i+1,j,k))*weights(ha(i+1,j,k))*abs(dscdx(inp)/normalized)**(weightsexp2) + &
              !   dscurv2(ha(i-1,j,k))*weights(ha(i-1,j,k))*abs(dscdx(inp)/normalized)**(weightsexp2) + &
              !   dscurv2(ha(i,j+1,k))*weights(ha(i,j+1,k))*abs(dscdy(inp)/normalized)**(weightsexp2) + &
              !   dscurv2(ha(i,j-1,k))*weights(ha(i,j-1,k))*abs(dscdy(inp)/normalized)**(weightsexp2) ) / &
              !  (weights(inp)+ &
              !   weights(ha(i+1,j,k))*abs(dscdx(inp)/normalized)**(weightsexp2) + &
              !   weights(ha(i-1,j,k))*abs(dscdx(inp)/normalized)**(weightsexp2) + &
              !   weights(ha(i,j+1,k))*abs(dscdy(inp)/normalized)**(weightsexp2) +  &
              !   weights(ha(i,j-1,k))*abs(dscdy(inp)/normalized)**(weightsexp2))
            else
              normalized=sqrt(dscdx(inp)**2+dscdy(inp)**2+dscdz(inp)**2)
              tempcurv2(inp)=dscurv2(inp)*weights(inp) + &
                (dscurv2(ha(i+1,j,k))*weights(ha(i+1,j,k))*abs(dscdx(inp)/normalized)**(weightsexp2)+ &
                 dscurv2(ha(i-1,j,k))*weights(ha(i-1,j,k))*abs(dscdx(inp)/normalized)**(weightsexp2)+ &
                 dscurv2(ha(i,j+1,k))*weights(ha(i,j+1,k))*abs(dscdy(inp)/normalized)**(weightsexp2)+ &
                 dscurv2(ha(i,j-1,k))*weights(ha(i,j-1,k))*abs(dscdy(inp)/normalized)**(weightsexp2)+ &
                 dscurv2(ha(i,j,k+1))*weights(ha(i,j,k+1))*abs(dscdz(inp)/normalized)**(weightsexp2)+ &
                 dscurv2(ha(i,j,k-1))*weights(ha(i,j,k-1))*abs(dscdz(inp)/normalized)**(weightsexp2)) / &
                (weights(inp) & 
                +weights(ha(i+1,j,k))*abs(dscdx(inp)/normalized)**(weightsexp2) &
                +weights(ha(i-1,j,k))*abs(dscdx(inp)/normalized)**(weightsexp2) &
                +weights(ha(i,j+1,k))*abs(dscdy(inp)/normalized)**(weightsexp2) &
                +weights(ha(i,j-1,k))*abs(dscdy(inp)/normalized)**(weightsexp2) &
                +weights(ha(i,j,k+1))*abs(dscdz(inp)/normalized)**(weightsexp2) &
                +weights(ha(i,j,k-1))*abs(dscdz(inp)/normalized)**(weightsexp2))
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
          if (dscurv2(inp)/=0d0) then
             dscurv2(inp)=tempcurv2(inp)
          end if
        end do
      end do
    end do
  end do
end if

if (nolocalcurv .eqv. .true.) then
  do m=1,nblo
    call setind(ngr,m, &
#               include "listsetind.h"            
                ier)            
    do k=2,nkm
      do i=2,nim
        do j=2,njm
          inp=ha(i,j,k)
           if (thirddim==1 .and. (xc(inp)<xclim1_3d .or. xc(inp)>xclim2_3d .or. zc(inp)<xclim1_3d .or. zc(inp)>xclim2_3d)) then
            dscurv2(inp)=0d0
          else if (thirddim==0 .and. (xc(inp)<xclim1 .or. xc(inp)>xclim2)) then
             dscurv2(inp)=0d0
           end if
        end do
      end do
    end do
  end do
end if

if (count1 > 0) then
  dscurv2a=dscurvsum2/count1
else 
  dscurv2a=0.d0
end if

if (iproc==1) then
  ! Activate the next line to see the average curvature in every iteration
!  write(*,1000) iter,lst,dscurv2a,count1
end if

1000 format('iter:',i10,'  Curv_method:',i2,'  Avrg_curv:',f8.2,'  N.vol in interface:',i7)

end subroutine curvature

!####################################################################################################################################
subroutine maxic(inp,dsabmax,&
#                include "listcalcurv.h"
                 ier)
!####################################################################################################################################
! Find max interface cell
!####################################################################################################################################
implicit none
#include "cb3dall.h"
#include "cbglobaldim.h"

! dummy variables
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

integer :: ier,icphi

! dummy variables
integer,intent(in) :: inp
real*8, intent(in out) :: dsabmax

! local variables
real*8 :: dsdeltaxi2

!####################################################################################################################################

if (c(inp) > 0.d0 .and. c(inp) < 1.d0) then
  ! Interface condition
  dsdeltaxi2=abs(x(inp)-x(inp-nj))
  if (dsabmax < dsdeltaxi2) then
     dsabmax=dsdeltaxi2
  end if

  dsdeltaxi2=abs(y(inp)-y(inp-1))  
  if (dsabmax < dsdeltaxi2) then
     dsabmax=dsdeltaxi2
  end if

  if (thirddim == 1) then
    dsdeltaxi2=abs(z(inp)-z(inp-nij)) 
    if (dsabmax < dsdeltaxi2) then
      dsabmax=dsdeltaxi2
    end if
  end if

end if

return
end subroutine maxic 

!##################################################################################################
subroutine chebyshev(ngr,filter,numa,numb,int1,out1,out2,out3,out4,lfy,lfend, &
#                    include "listcalcurv.h"
                     ier)
!##################################################################################################
!Chebyshev Filter Function 
!##################################################################################################
implicit none
#include "cb3dall.h"
#include "cbglobaldim.h"

! dummy variables
integer, intent(in) :: ngr
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

logical, intent (in) :: lfy,lfend
integer,intent(in)  :: numa,numb,filter
real*8, intent (in)  :: int1(*) 
real*8, intent(out)  :: out1(*),out2(*),out3(*),out4(*) 
integer:: ier

!Local variables
integer :: m,k,j,i,inp,count1
integer :: k1,kn,j1,jn,i1,inn
real*8  :: help1(nxyza),help2(nxyza),help3(nxyza),help4(nxyza)
real*8  :: dscurvsum2,dscurv2a
real*8  :: ch1,ch2,ch3,ch4
real*8  :: extrapo_cx,extrapo_cy,extrapo_cz,extrapo_xc,extrapo_yc,extrapo_zc

#include "hahihjhk.h" 
!##################################################################################################
call setzero(help1,nxyza,1,nxyza,ier)
call setzero(help2,nxyza,1,nxyza,ier)
call setzero(help3,nxyza,1,nxyza,ier)
call setzero(help4,nxyza,1,nxyza,ier)
call setzero(out1,nxyza,1,nxyza,ier)
call setzero(out2,nxyza,1,nxyza,ier)
call setzero(out3,nxyza,1,nxyza,ier)
call setzero(out4,nxyza,1,nxyza,ier)

! Filter coefficients as mentioned in Palacio(2002)-Smoothing of digital...
select case (filter)
  ! LSP Degree 0
  case(1)
    ch1=1.d0
    ch2=1.d0
    ch3=1.d0
  ! Binominal 
  case (2)
    ch1=1d0
    ch2=4d0
    ch3=6d0
  ! 5-Point Chebyshev 
  case (3)
    ch1=-1.d0
    ch2=4.d0
    ch3=10.d0
  ! LSP Degree 2
  case(4)
    ch1=-3.d0
    ch2=12.d0
    ch3=17.d0
end select
! denominator is sum (a*ch1+b*ch2+c*ch3+d*ch2+e*ch1/sum(ch1+ch2+ch3+ch2+ch1))
ch4=2.d0*ch1+2.d0*ch2+1.d0*ch3

dscurvsum2=0.d0
count1=0

do m=1,nblo
  call setind(ngr,m, &
#             include"listsetind.h"
              ier)

  j1=2
  jn=njm
  i1=2
  inn=nim
  k1=2
  if (thirddim == 1) then
    kn=nkm
    if (lfy) then
      j1=4
      jn=njm-2
      i1=4
      inn=nim-2
      k1=4
      kn=nkm-2
    end if
  else if (thirddim == 0) then
    kn=2
    if (lfy) then
      j1=4
      jn=njm-2
      i1=4
      inn=nim-2
    end if
  end if

  do k=k1,kn
    do i=i1,inn 
      do j=j1,jn
        inp=ha(i,j,k)
        ! i-direction
        if (i < numa ) then
          ! extrapolate x-coordinate 
          extrapo_xc=xc(inp-1*nj) - ( (xc(inp+1*nj)-xc(inp)) + (xc(inp)-xc(inp-1*nj)) )/2
          ! extrapolate value
          extrapo_cx=int1(inp-1*nj) + ( (extrapo_xc-xc(inp-1*nj)) / (xc(inp) - xc(inp-1*nj)) ) &
                                    * (int1(inp) - int1(inp-1*nj))
          extrapo_cx=min(max(extrapo_cx,0.d0),1.d0)
          help1(inp)=(ch1*extrapo_cx+ch2*int1(inp-1*nj)+ch3*int1(inp)+ch2*int1(inp+1*nj) &
                           +ch1*int1(inp+2*nj))/ch4
        else if (i > (nim-numb) ) then
          ! extrapolate x-coordinate 
          extrapo_xc=xc(inp+1*nj) + ( (xc(inp+1*nj)-xc(inp)) + (xc(inp)-xc(inp-1*nj)) )/2
          ! extrapolate value
          extrapo_cx=int1(inp+1*nj) + ( (extrapo_xc-xc(inp+1*nj)) / (xc(inp) - xc(inp+1*nj)) ) &
                                          * (int1(inp) - int1(inp-1*nj))
          extrapo_cx=min(max(extrapo_cx,0.d0),1.d0)
          help1(inp)=(ch1*int1(inp-2*nj)+ch2*int1(inp-1*nj)+ch3*int1(inp)+ch2*int1(inp+1*nj)&
                           +ch1*extrapo_cx)/ch4
        else
          help1(inp)=(ch1*int1(inp-2*nj)+ch2*int1(inp-1*nj)+ch3*int1(inp)+ch2*int1(inp+1*nj)+ch1*int1(inp+2*nj))/ch4
        end if

        ! j-direction
        if (j < numa ) then
          ! extrapolate y-coordinate 
          extrapo_yc=yc(inp-1) - ( (yc(inp+1)-yc(inp)) + (yc(inp)-yc(inp-1)) )/2
          ! extrapolate value
          extrapo_cy=int1(inp-1) + ( (extrapo_yc-yc(inp-1)) / (yc(inp) - yc(inp-1)) ) * (int1(inp) - int1(inp-1))
          extrapo_cy=min(max(extrapo_cy,0.d0),1.d0)
          help2(inp)=(ch1*extrapo_cy+ch2*int1(inp-1)+ch3*int1(inp)+ch2*int1(inp+1)&
                           +ch1*int1(inp+2))/ch4
        else if (j > (njm-numb) ) then
          ! extrapolate y-coordinate 
          extrapo_yc=yc(inp+1) + ( (yc(inp+1)-yc(inp)) + (yc(inp)-yc(inp-1)) )/2
          ! extrapolate value
          extrapo_cy=int1(inp+1) + ( (extrapo_yc-yc(inp+1)) / (yc(inp) - yc(inp+1)) ) * (int1(inp) - int1(inp+1))
          extrapo_cy=min(max(extrapo_cy,0.d0),1.d0)
          help2(inp)=(ch1*int1(inp-2)+ch2*int1(inp-1)+ch3*int1(inp)+ch2*int1(inp+1)&
                           +ch1*extrapo_cy)/ch4
        else
          help2(inp)=(ch1*int1(inp-2)+ch2*int1(inp-1)+ch3*int1(inp)+ch2*int1(inp+1)+ch1*int1(inp+2))/ch4
        end if

        if (thirddim == 0) then
          help3(inp)=0.d0   
          help4(inp)=0.5d0*(help1(inp)+help2(inp))
        else
          ! k-direction
          if (k < numa) then
            ! extrapolate z-coordinate 
            extrapo_zc=zc(inp-1*nij) - ( (zc(inp+1*nij)-zc(inp)) + (zc(inp)-zc(inp-1*nij)) )/2
            ! extrapolate value
            extrapo_cz=int1(inp-1*nij) + ( (extrapo_zc-zc(inp-1*nij)) / (zc(inp) - zc(inp-1*nij)) ) * (int1(inp) - int1(inp-1*nij))
            extrapo_cz=min(max(extrapo_cz,0.d0),1.d0)
            help3(inp)=(ch1*extrapo_cz+ch2*int1(inp-1*nij)+ch3*int1(inp)+ch2*int1(inp+1*nij)&
                             +ch1*int1(inp+2*nij))/ch4
          else if (k > nkm-numb) then
            ! extrapolate z-coordinate 
            extrapo_zc=zc(inp+1*nij) + ( (zc(inp+1*nij)-zc(inp)) + (zc(inp)-zc(inp-1*nij)) )/2
            ! extrapolate value
            extrapo_cz=int1(inp+1*nij) + ( (extrapo_zc-zc(inp+1*nij)) / (zc(inp) - zc(inp+1*nij)) ) * (int1(inp) - int1(inp+1*nij))
            extrapo_cz=min(max(extrapo_cz,0.d0),1.d0)
            help3(inp)=(ch1*int1(inp-2*nij)+ch2*int1(inp-1*nij)+ch3*int1(inp)+ch2*int1(inp+1*nij)&
                             +ch1*extrapo_cz)/ch4
          else
            help3(inp)=(ch1*int1(inp-2*nij)+ch2*int1(inp-1*nij)+ch3*int1(inp)+ch2*int1(inp+1*nij)+ &
                       ch1*int1(inp+2*nij))/ch4
          end if 
            ! Smoothing like Park (25) 
            help4(inp)=1.d0/3.d0*(help1(inp)+help2(inp)+help3(inp))
        end if     

        if ((lfend).and.(int1(inp)/=0)) then
          dscurvsum2=dscurvsum2+help4(inp)
          count1=count1+1
        end if
 
      end do !j
    end do !i
  end do !k
end do !m

do m=1,nblo
  call setind(ngr,m, &
#             include"listsetind.h"
              ier)

  j1=2
  jn=njm
  i1=2
  inn=nim
  k1=2

  if (thirddim == 1) then
    kn=nkm
    if (lfy) then
      j1=4
      jn=njm-2
      i1=4
      inn=nim-2
      k1=4
      kn=nkm-2
    end if
  else if (thirddim == 0) then
    kn=2
    if (lfy) then
      j1=4
      jn=njm-2
      i1=4
      inn=nim-2
    end if
  end if

  do k=k1,kn
    do i=i1,inn 
      do j=j1,jn
        inp=ha(i,j,k)
        out1(inp)=help1(inp)
        out2(inp)=help2(inp)
        out3(inp)=help3(inp)
        out4(inp)=help4(inp)
      end do !j
    end do !i
  end do !k
end do !m

if (lfend) then
  call sumbro(dscurvsum2,ier)
  call sumbro(count1,ier)

  if (count1 > 0) then
    dscurv2a=dscurvsum2/count1
  else 
    dscurv2a=0.d0
  end if

  if (iproc==1) then
    ! Activate the next line to see the average curvature in every iteration
!    write(*,1000) iter,lst,dscurv2a,count1
  end if
end if 

1000 format('iter:',i10,'  Curv_method:',i2,'  Avrg_curv:',f8.2,'  N.vol in interface:',i7)

end subroutine chebyshev

!##################################################################################################
subroutine chebyshev2(ngr,icphi,filter,numa,numb,int1,out1,out2,out3,out4,lfy,lfend, &
#                    include "listcalcurv.h"
                     ier)
!##################################################################################################
!Chebyshev Filter Function 
!##################################################################################################
implicit none
#include "cb3dall.h"
#include "cbglobaldim.h"

! dummy variables
integer, intent(in) :: ngr
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

logical, intent (in)  :: lfy,lfend
integer, intent (in)  :: numa,numb,filter,icphi

real*8, intent (in)   :: int1(*) 
real*8, intent (out)  :: out1(*),out2(*),out3(*),out4(*) 
integer:: ier

!Local variables
integer :: m,k,j,i,inp,count1,incst,incst1
integer :: k1,kn,j1,jn,i1,inn
real*8  :: help1(2*nxyza),help2(2*nxyza),help3(2*nxyza),help4(2*nxyza)
real*8  :: dscurvsum2,dscurv2a
real*8  :: ch1,ch2,ch3,ch4
real*8  :: extrapo_cx,extrapo_cy,extrapo_cz,extrapo_xc,extrapo_yc,extrapo_zc

#include "hahihjhk.h" 
!##################################################################################################
incst=icphi*nxyza
incst1=1+icphi*nxyza

! Set zero first from 1 to nxyza and then from nxyza+1 to 2*nxyza
call setzero(help1,2*nxyza,incst1,(icphi+1)*nxyza,ier)
call setzero(help2,2*nxyza,incst1,(icphi+1)*nxyza,ier)
call setzero(help3,2*nxyza,incst1,(icphi+1)*nxyza,ier)
call setzero(help4,2*nxyza,incst1,(icphi+1)*nxyza,ier)
call setzero(out1,2*nxyza,incst1,(icphi+1)*nxyza,ier)
call setzero(out2,2*nxyza,incst1,(icphi+1)*nxyza,ier)
call setzero(out3,2*nxyza,incst1,(icphi+1)*nxyza,ier)
call setzero(out4,2*nxyza,incst1,(icphi+1)*nxyza,ier)



do m=1,nblo
  call setind(ngr,m, &
#             include"listsetind.h"
              ier)

  do k=2,nkm
    do i=2,nim
      do j=2,njm
        inp=ha(i,j,k)
        out4(inp+incst)=int1(inp+incst)
      end do !j
    end do !i
  end do !k
end do

call exall(ngr,out4,1,1, &
#            include "listexall.h"
             ier)

! Filter coefficients as mentioned in Palacio(2002)-Smoothing of digital...
select case (filter)
  ! LSP Degree 0
  case(1)
    ch1=1.d0
    ch2=1.d0
    ch3=1.d0
  ! Binominal 
  case (2)
    ch1=1d0
    ch2=4d0
    ch3=6d0
  ! 5-Point Chebyshev 
  case (3)
    ch1=-1.d0
    ch2=4.d0
    ch3=10.d0
  ! LSP Degree 2
  case(4)
    ch1=-3.d0
    ch2=12.d0
    ch3=17.d0
end select
! denominator is sum (a*ch1+b*ch2+c*ch3+d*ch2+e*ch1/sum(ch1+ch2+ch3+ch2+ch1))
ch4=2.d0*ch1+2.d0*ch2+1.d0*ch3


dscurvsum2=0.d0
count1=0

do m=1,nblo
  call setind(ngr,m, &
#             include"listsetind.h"
              ier)

  j1=2
  jn=njm
  i1=2
  inn=nim
  k1=2

  if (thirddim == 1) then
    kn=nkm
    if (lfy) then
      j1=4
      jn=njm-2
      i1=4
      inn=nim-2
      k1=4
      kn=nkm-2
    end if
  else if (thirddim == 0) then
    kn=2
    if (lfy) then
      j1=4
      jn=njm-2
      i1=4
      inn=nim-2
    end if
  end if

  do k=k1,kn
    do i=i1,inn 
      do j=j1,jn
        inp=ha(i,j,k)
        ! i-direction
        if (i < numa ) then
          ! extrapolate x-coordinate 
          extrapo_xc=xc(inp-1*nj) - ( (xc(inp+1*nj)-xc(inp)) + (xc(inp)-xc(inp-1*nj)) )/2
          ! extrapolate value
          extrapo_cx=int1(inp+incst-1*nj) + ( (extrapo_xc-xc(inp-1*nj)) / (xc(inp) - xc(inp-1*nj)) ) &
                                          * (int1(inp+incst) - int1(inp+incst-1*nj))
          extrapo_cx=min(max(extrapo_cx,0.d0),1.d0)
          help1(inp+incst)=(ch1*extrapo_cx+ch2*int1(inp+incst-1*nj)+ch3*int1(inp+incst)+ch2*int1(inp+incst+1*nj)&
                           +ch1*int1(inp+incst+2*nj))/ch4
        else if (i > (nim-numb) ) then
          ! extrapolate x-coordinate 
          extrapo_xc=xc(inp+1*nj) + ( (xc(inp+1*nj)-xc(inp)) + (xc(inp)-xc(inp-1*nj)) )/2
          ! extrapolate value
          extrapo_cx=int1(inp+incst+1*nj) + ( (extrapo_xc-xc(inp+1*nj)) / (xc(inp) - xc(inp+1*nj)) ) &
                                          * (int1(inp+incst) - int1(inp+incst-1*nj))
          extrapo_cx=min(max(extrapo_cx,0.d0),1.d0)
          help1(inp+incst)=(ch1*int1(inp+incst-2*nj)+ch2*int1(inp+incst-1*nj)+ch3*int1(inp+incst)+ch2*int1(inp+incst+1*nj)&
                           +ch1*extrapo_cx)/ch4
        else
          help1(inp+incst)=(ch1*int1(inp+incst-2*nj)+ch2*int1(inp+incst-1*nj)+ch3*int1(inp+incst)&
                                                    +ch2*int1(inp+incst+1*nj)+ch1*int1(inp+incst+2*nj))/ch4
        end if

        if (j < numa ) then
          ! extrapolate y-coordinate 
          extrapo_yc=yc(inp-1) - ( (yc(inp+1)-yc(inp)) + (yc(inp)-yc(inp-1)) )/2
          ! extrapolate value
          extrapo_cy=int1(inp+incst-1) + ( (extrapo_yc-yc(inp-1)) / (yc(inp) - yc(inp-1)) ) * (int1(inp+incst) - int1(inp+incst-1))
          extrapo_cy=min(max(extrapo_cy,0.d0),1.d0)
          help2(inp+incst)=(ch1*extrapo_cy+ch2*int1(inp+incst-1)+ch3*int1(inp+incst)+ch2*int1(inp+incst+1)&
                           +ch1*int1(inp+incst+2))/ch4
        else if (j > (njm-numb) ) then
          ! extrapolate y-coordinate 
          extrapo_yc=yc(inp+1) + ( (yc(inp+1)-yc(inp)) + (yc(inp)-yc(inp-1)) )/2
          ! extrapolate value
          extrapo_cy=int1(inp+incst+1) + ( (extrapo_yc-yc(inp+1)) / (yc(inp) - yc(inp+1)) ) * (int1(inp+incst) - int1(inp+incst+1))
          extrapo_cy=min(max(extrapo_cy,0.d0),1.d0)
          help2(inp+incst)=(ch1*int1(inp+incst-2)+ch2*int1(inp+incst-1)+ch3*int1(inp+incst)+ch2*int1(inp+incst+1)&
                           +ch1*extrapo_cy)/ch4
        else
          help2(inp+incst)=(ch1*int1(inp+incst-2)+ch2*int1(inp+incst-1)+ch3*int1(inp+incst) &
                                                 +ch2*int1(inp+incst+1)+ch1*int1(inp+incst+2))/ch4
        end if

        if (thirddim == 0) then
          help3(inp+incst)=0.d0   
          help4(inp+incst)=0.5d0*(help1(inp+incst)+help2(inp+incst))
        
        else if (thirddim == 1) then
          ! k-direction
          if (k < numa) then
            ! extrapolate z-coordinate 
            extrapo_zc=zc(inp-1*nij) - ( (zc(inp+1*nij)-zc(inp)) + (zc(inp)-zc(inp-1*nij)) )/2
            ! extrapolate value
            extrapo_cz=int1(inp+incst-1*nij) + ( (extrapo_zc-zc(inp-1*nij)) / (zc(inp) - zc(inp-1*nij)) ) &
                                               * (int1(inp+incst) - int1(inp+incst-1*nij))
            extrapo_cz=min(max(extrapo_cz,0.d0),1.d0)
            help3(inp+incst)=(ch1*extrapo_cz+ch2*int1(inp+incst-1*nij)+ch3*int1(inp+incst)+ch2*int1(inp+incst+1*nij)&
                             +ch1*int1(inp+incst+2*nij))/ch4
          else if (k > (nkm-numb)) then
            ! extrapolate z-coordinate 
            extrapo_zc=zc(inp+1*nij) + ( (zc(inp+1*nij)-zc(inp)) + (zc(inp)-zc(inp-1*nij)) )/2
            ! extrapolate value
            extrapo_cz=int1(inp+incst+1*nij) + ( (extrapo_zc-zc(inp+1*nij)) / (zc(inp) - zc(inp+1*nij)) ) &
                                              * (int1(inp+incst) - int1(inp+incst+1*nij))
            extrapo_cz=min(max(extrapo_cz,0.d0),1.d0)
            help3(inp+incst)=(ch1*int1(inp+incst-2*nij)+ch2*int1(inp+incst-1*nij)+ch3*int1(inp+incst)+ch2*int1(inp+incst+1*nij)&
                             +ch1*extrapo_cz)/ch4
          else
            help3(inp+incst)=(ch1*int1(inp+incst-2*(nij))+ch2*int1(inp+incst-1*(nij))+ch3*int1(inp+incst) &
                                                         +ch2*int1(inp+incst+1*(nij))+ch1*int1(inp+incst+2*(nij)))/ch4
          end if 
            ! Smoothing like Park (25) 
            help4(inp+incst)=1.d0/3.d0*(help1(inp+incst)+help2(inp+incst)+help3(inp+incst))
        end if     

        if ((lfend).and.(int1(inp)/=0)) then
          dscurvsum2=dscurvsum2+help4(inp)
          count1=count1+1
        end if
 
      end do !j
    end do !i
  end do !k
end do !m

do m=1,nblo
  call setind(ngr,m, &
#             include"listsetind.h"
              ier)

  j1=2
  jn=njm
  i1=2
  inn=nim
  k1=2
  if (thirddim == 1) then
    kn=nkm
    if (lfy) then
      j1=4
      jn=njm-2
      i1=4
      inn=nim-2
      k1=4
      kn=nkm-2
    end if
  else if (thirddim == 0) then
    kn=2
    if (lfy) then
      j1=4
      jn=njm-2
      i1=4
      inn=nim-2
    end if
  end if

  do k=k1,kn
    do i=i1,inn 
      do j=j1,jn
        inp=ha(i,j,k)
        out1(inp+incst)=help1(inp+incst)
        out2(inp+incst)=help2(inp+incst)
        out3(inp+incst)=help3(inp+incst)
        out4(inp+incst)=help4(inp+incst)
      end do !j
    end do !i
  end do !k
end do !m

if (lfend) then
  if (count1 > 0) then
    dscurv2a=dscurvsum2/count1
  else 
    dscurv2a=0.d0
  end if
  if (iproc==1) then
    ! Activate the next line to see the average curvature in every iteration
!    write(*,1000) iter,lst,dscurv2a,count1
  end if
end if 

1000 format('iter:',i10,'  Curv_method:',i2,'  Avrg_curv:',f8.2,'  N.vol in interface:',i7)
end subroutine chebyshev2

