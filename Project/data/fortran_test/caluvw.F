#include "listdefines.h"
!###################################################################################################################################
subroutine caluvw(ngr, &
#                 include "listcaluvw.h"
                  ier)
!###################################################################################################################################
!  Calculate velocity components u,v, and w
!###################################################################################################################################
use initRigidBodyMotion_mod
implicit none
#include "cb3dall.h"
#include "cbglobaldim.h"

! dummy variables
integer, intent(in out) :: ngr
#include "dimacoef.h"
#include "dimbcoef.h"
#include "dimlilk.h"
#include "dimliglkg.h"
#include "dimindex2.h"
#include "dimiters.h"
#include "dimindex4.h"
#include "dimlogic4.h"
#include "dimdivers.h"
!#include "dimdslog.h" !
#include "dimstoffw12.h"
#include "dimifldir.h"
#include "dimnuamol.h"
#include "dimuvwvel.h"
#include "dimuvwvelg.h"
#include "dimuvwvold.h"
#include "dimtemp.h"
#include "dimtempg.h"
#include "dimtempold.h"
#include "dimkeps.h"
#include "dimkepsg.h"
#include "dimkepsold.h"
#include "dimlowre.h"
#include "dimgeom12.h"
#include "dimgeomm123.h"
#include "dimgeomt12.h"
#include "dimbndcon.h"
#include "dimbndcont.h"
#include "dimrhelp3.h"
#include "dimblopar.h"
#include "dimtayint.h"
#include "dimcalgeomcel1.h"
#include "dimcalgeomcel2.h"
#include "dimcalgeomcel3.h"
#include "dimhybblen.h"
#include "dimvelgrad.h"
#include "dimconc.h"
#include "dimconcold.h"
#include "dimuvwvof.h"
#include "dimdiscf.h"
#include "dimcvof.h"
#include "dimdsls.h"
#include "dimdsst.h"
#include "dimdscurv.h"
#include "dimcderivatives.h"

integer :: ier

! local parameters
real*8, parameter :: pi=3.1415926535897932384626433832795028841971693d0

! local variables
integer :: m,ivec,inp,nie,nje,nke,int,inb,inn,ins,ine,inw,inbs,inbw,inbsw,insw,inwb,idew,idns,idtb,insb
real*8 :: rtidis,dtidis,urfru,urfmu,urfrv,urfmv,urfrw,urfmw
real*8 :: heat,press,densit,apall,aptu,aptv,aptw
real*8 :: sumu,sumv,sumw,apo,suc,svc,swc
real*8 :: fden,fbeta,fden2
real*8 :: grx,gry,grz
real*8 :: dxks,dyks,dzks,dxet,dyet,dzet,dxzd,dyzd,dzzd,detJ
real*8 :: j11,j12,j13,j21,j22,j23,j31,j32,j33
integer :: i,j,k

! Variables for VOF filtering
! logical, intent(in) :: lcon,lmix
! integer,intent(in) :: ngr
! real*8,  intent(out) :: dscurv2a
! integer:: ier
logical :: lfy
integer :: incst1,icphi,order
logical :: concfiltering,concfiltering2nd,lfend
integer :: numa,numb,concfilter,concfilter2nd
real*8  :: fi1(2*nxyza),fi2(2*nxyza),fi3(2*nxyza),fi4(2*nxyza),fi4cdx(2*nxyza),fi4cdy(2*nxyza),fi4cdz(2*nxyza)
integer :: incst

#include "hahihjhk.h"
!###################################################################################################################################

rtidis = 1.d0 / ftidis
dtidis = 1.d0 - ftidis

call setval(ngr,be,0.d0, &
#           include "listsetval.h"
            ier)
call setval(ngr,bw,0.d0, &
#           include "listsetval.h"
            ier)
call setval(ngr,bt,0.d0, &
#           include "listsetval.h"
            ier)
call setval(ngr,bb,0.d0, &
#           include "listsetval.h"
            ier)
call setval(ngr,bn,0.d0, &
#           include "listsetval.h"
            ier)
call setval(ngr,bs,0.d0, &
#           include "listsetval.h"
            ier)
call setval(ngr,bp,0.d0, &
#           include "listsetval.h"
            ier)
call setval(ngr,res,0.d0, &
#           include "listsetval.h"
            ier)
call setval(ngr,gxm,0.d0, &
#           include "listsetval.h"
            ier)
call setval(ngr,gym,0.d0, &
#           include "listsetval.h"
            ier)
call setval(ngr,gzm,0.d0, &
#           include "listsetval.h"
            ier)

! Calculate pressure gradients
call calcdp(ngr, &
#           include "listcalcdp.h"
            ier)

urfru=1.d0/urf(iu,ngr)
urfmu=1.d0-urf(iu,ngr)
urfrv=1.d0/urf(iv,ngr)
urfmv=1.d0-urf(iv,ngr)
urfrw=1.d0/urf(iw,ngr)
urfmw=1.d0-urf(iw,ngr)

! Start VOF calculation
if(lconc)then
  do m=1,nblo
    if (lfluidr(nfr(m))) then ! fluid region
      call setind(ngr,m, &
#                 include "listsetind.h"
                  ier)
          do inp=icst,icen
            fpx(inp) = 0.
            fpy(inp) = 0.
            fpz(inp) = 0.
          end do
    end if
  end do

! Overwrite dsdirac with 1 and recalculate dscdx/y/z with calgrad

! Option for concentration filter
concfiltering=.false.  ! war false
concfiltering2nd=.false.
!1-lsp0,2-binominal,3-5pointcheby,4-lsp2
concfilter=4
concfilter2nd=4
! Determine the order of the Gradients [1-first order, 2-second order with first order at borders]
order=1
! Filter concentration only for gradient computation
if (concfiltering .eqv. .true.) then
  do icphi=0,mcphi
    incst=icphi*nxyza
    incst1=incst+1
    call exall(ngr,c(incst1),1,1, &
#            include "listexall.h"
             ier)
  end do
  numa=3
  numb=1
  lfy=.false.
  lfend=.false.
  do icphi=0,mcphi
    call chebyshev2(ngr,icphi,concfilter,numa,numb,c,fi1,fi2,fi3,fi4,lfy,lfend, &
! #                  include "listcalcurv.h"
#                  include "listcaluvw.h"
                   ier)
  end do
  !Gradients in cell center; input cell center
  call calgradc(ngr,0,1,fi4,dscdx,dscdy,dscdz, &
#                   include "listcalgradc.h"
                    ier)
else
    !Gradients in cell center; input cell center
    call calgradc(ngr,0,1,c,dscdx,dscdy,dscdz, &
#                   include "listcalgradc.h"
                    ier)
end if
call bconphi(dscdx,ngr, &
#          include "listbpres.h"
           ier)
call bconphi(dscdy,ngr, &
#          include "listbpres.h"
           ier)
call bconphi(dscdz,ngr, &
#          include "listbpres.h"
           ier)
call exalv(ngr,dscdx,dscdy,dscdz,1,1,1,&
#          include "listexalv.h"
           ier)

do m=1,nblo
  call setind(ngr,m, &
#             include "listsetind.h"
              ier)
  do k=1,nk
    do i=1,ni
      do j=1,nj
        inp=ha(i,j,k)
        dsdirac(inp) = 1
      end do
    end do
  end do
end do

do m=1,nblo
  call setind(ngr,m, &
#             include "listsetind.h"
              ier)
  do k=1,nk
    do i=1,ni
      do j=1,nj
        inp=ha(i,j,k)
        
        fpx(inp)=gravx*den(inp)
        fpy(inp)=gravy*den(inp) 
        fpz(inp)=gravz*den(inp)

        ! Add surface tension calculation to volume force
        fstx(inp)=dskappa*dscurv2(inp)*dscdx(inp)*dsdirac(inp)
         fpx(inp)=fpx(inp)+fstx(inp)
        
         ! print*, dscdx(inp), cdx(inp)
                        
        fsty(inp)=dskappa*dscurv2(inp)*dscdy(inp)*dsdirac(inp)
         fpy(inp)=fpy(inp)+fsty(inp)

        fstz(inp)=dskappa*dscurv2(inp)*dscdz(inp)*dsdirac(inp)
         fpz(inp)=fpz(inp)+fstz(inp)

      end do
    end do
  end do
end do

! Exchange volume force between blocks
call exall(ngr,fpx,1,1, &
#          include "listexall.h"
           ier)
call exall(ngr,fpy,1,1, &
#          include "listexall.h"
           ier)
call exall(ngr,fpz,1,1, &
#          include "listexall.h"
           ier)
         

! Calculate volume forces (incl surface tension) on faces -> be, bt, bn in calcc.F
call meanb3(ngr,0,1, &
#           include "listmeanb.h"
            ier)


! Exchange be,bt,bn between blocks
call exall(ngr,be,1,1, &
#          include "listexall.h"
           ier)
call exall(ngr,bn,1,1, &
#          include "listexall.h"
           ier)
call exall(ngr,bt,1,1, &
#          include "listexall.h"
           ier)

! Interpolation of the volume forces for the cells
! Notice taht gxm,gym,gzm contain vol(inp) discretization as in the case of the pressure gradient
do m=1,nblo
  call setind(ngr,m, &
#             include "listsetind.h"
              ier)
          
  idew = nj
  idns = 1
  idtb = nij

  do k=2,nkm
    do i=2,nim
      do j=2,njm
        inp=ha(i,j,k)
        inw=inp-idew
        ine=inp+idew
        ins=inp-idns
        inn=inp+idns
        inb=inp-idtb
        int=inp+idtb
        insb=ins-idtb
        insw=ins-idew
        inwb=inw-idtb
        inbs=inb-1
        insw=inw-1
        inbw=inw-nij
        inbsw=inbw-1

        dxks=0.25d0*(x(inp)+x(ins )+x(inb )+x(inbs ) -x(inw)-x(insw)-x(inbw)-x(inbsw))
        dyks=0.25d0*(y(inp)+y(ins )+y(inb )+y(inbs ) -y(inw)-y(insw)-y(inbw)-y(inbsw))
        dzks=0.25d0*(z(inp)+z(ins )+z(inb )+z(inbs ) -z(inw)-z(insw)-z(inbw)-z(inbsw))
        
        dxet=0.25d0*(x(inp)+x(inb )+x(inw )+x(inbw ) -x(ins)-x(inbs)-x(insw)-x(inbsw))
        dyet=0.25d0*(y(inp)+y(inb )+y(inw )+y(inbw ) -y(ins)-y(inbs)-y(insw)-y(inbsw))
        dzet=0.25d0*(z(inp)+z(inb )+z(inw )+z(inbw ) -z(ins)-z(inbs)-z(insw)-z(inbsw))
        
        dxzd=0.25d0*(x(inp)+x(ins )+x(inw )+x(insw ) -x(inb)-x(inbs)-x(inbw)-x(inbsw))
        dyzd=0.25d0*(y(inp)+y(ins )+y(inw )+y(insw ) -y(inb)-y(inbs)-y(inbw)-y(inbsw))
        dzzd=0.25d0*(z(inp)+z(ins )+z(inw )+z(insw ) -z(inb)-z(inbs)-z(inbw)-z(inbsw))
               
        j11=dxks/sqrt(dxks**2+dyks**2+dzks**2)
        j12=dyks/sqrt(dxks**2+dyks**2+dzks**2)
        j13=dzks/sqrt(dxks**2+dyks**2+dzks**2)
        
        j21=dxet/sqrt(dxet**2+dyet**2+dzet**2)
        j22=dyet/sqrt(dxet**2+dyet**2+dzet**2)
        j23=dzet/sqrt(dxet**2+dyet**2+dzet**2)
        
        j31=dxzd/sqrt(dxzd**2+dyzd**2+dzzd**2)
        j32=dyzd/sqrt(dxzd**2+dyzd**2+dzzd**2)
        j33=dzzd/sqrt(dxzd**2+dyzd**2+dzzd**2)

        ! detJ =1, if equidistant and detJ=vol,if there is no norming sqrt(dxks**2+dyks**2+dzks**2)
        detJ=j11*j22*j33+j12*j23*j31+j13*j21*j32-j13*j22*j31-j12*j21*j33-j11*j32*j23
        ! detJ=1.d0
       
        ! gxm is volume force in (inp), calculated with forces on faces (be,bn,bt)
        ! Koefficiants are the same, difference comes from (inp),(inw), etc. -- be <-> bw; bn <-> bs; bt <-> bb
        gxm(inp)= (fx(inp)*be(inp) + (1.-fx(inw))*be(inw)) *ar1x(inp)/detJ &
                 +(fy(inp)*bn(inp) + (1.-fy(ins))*bn(ins)) *ar2x(inp)/detJ &
                 +(fz(inp)*bt(inp) + (1.-fz(inb))*bt(inb)) *ar3x(inp)/detJ

        gym(inp)= (fx(inp)*be(inp) + (1.-fx(inw))*be(inw)) *ar1y(inp)/detJ &
                 +(fy(inp)*bn(inp) + (1.-fy(ins))*bn(ins)) *ar2y(inp)/detJ &
                 +(fz(inp)*bt(inp) + (1.-fz(inb))*bt(inb)) *ar3y(inp)/detJ
             
        gzm(inp)= (fx(inp)*be(inp) + (1.-fx(inw))*be(inw)) *ar1z(inp)/detJ &
                 +(fy(inp)*bn(inp) + (1.-fy(ins))*bn(ins)) *ar2z(inp)/detJ &
                 +(fz(inp)*bt(inp) + (1.-fz(inb))*bt(inb)) *ar3z(inp)/detJ
      end do
    end do
  end do
         
  do inp=icst,icen
    be(inp)=0.d0
    bn(inp)=0.d0
    bt(inp)=0.d0
    bw(inp)=0.d0
    bs(inp)=0.d0
    bb(inp)=0.d0           
  end do

end do

call exalv(ngr,gxm,gym,gzm,1,1,1, &
#          include "listexalv.h"
           ier)

! End VOF calculation
endif



do  m=1,nblo
  if (lfluidr(nfr(m))) then
    call setind(ngr,m, &
#               include "listsetind.h"
                ier)

    ! Calculate source terms integrated over volume
    heat  = 0.d0
    press = 0.d0
    
    densit=fden(press,tf(nfr(m)),nfr(m))
    
!$omp parallel default(shared), private(ivec,inp,heat)
    if (.not.loldfl) then
#include "vdirnodep.h"
!$omp do schedule(static)
      do ivec = iis,iie
        inp=ijkvec(ivec)
        su(inp) = (-ar1x(inp)*dp1(inp) - ar2x(inp)*dp2(inp) - ar3x(inp)*dp3(inp)) * rtidis
        sv(inp) = (-ar1y(inp)*dp1(inp) - ar2y(inp)*dp2(inp) - ar3y(inp)*dp3(inp)) * rtidis
        sw(inp) = (-ar1z(inp)*dp1(inp) - ar2z(inp)*dp2(inp) - ar3z(inp)*dp3(inp)) * rtidis
      end do
!$omp end do
    else
#include "vdirnodep.h"
!$omp do schedule(static)
      do ivec = iis,iie
        inp=ijkvec(ivec)
        su(inp) = 0.d0
        sv(inp) = 0.d0
        sw(inp) = 0.d0
      end do
!$omp end do
    end if


    ! Add gravitational acceleration (+surface tension) to the source terms
    do ivec = iis,iie
      inp=ijkvec(ivec)
      if(lconc)then
        ! Special discretization rule
        grx = gxm(inp)*rtidis 
        gry = gym(inp)*rtidis 
        grz = gzm(inp)*rtidis 
      else
        ! Midpoint rule
        grx = gravx * den(inp) * vol(inp) * rtidis
        gry = gravy * den(inp) * vol(inp) * rtidis
        grz = gravz * den(inp) * vol(inp) * rtidis 
      endif        
           
      su(inp) = su(inp) + grx 
      sv(inp) = sv(inp) + gry
      sw(inp) = sw(inp) + grz
    enddo

    ! Calculate contribution from time terms moved from the prepstep.F routine, because in VOF density is variable
    ! Only for itidis=1 - fofi
    if (lconc) then
      if (lcal(iu)) then
        if (lmvgr) then
          do ivec=iis,iie
            inp=ijkvec(ivec)
            ! uold, because from the last timestep; in prepstep it is with u, because there uold=u hasn't been set 
            sut(inp)=den(inp)*volold(inp)*dtr*uold(inp)
            svt(inp)=den(inp)*volold(inp)*dtr*vold(inp)
            swt(inp)=den(inp)*volold(inp)*dtr*wold(inp)
            ! volold is been set before prepstep call in unsteady, therefor volold also in prepstep
          end do
          if (itidis.eq.2) then
            STOP 'caluvw.F: loldtime not existent here'
          end if
        else
          do ivec=iis,iie
            inp=ijkvec(ivec)
            sut(inp)=den(inp)*vol(inp)*dtr*uold(inp)
            svt(inp)=den(inp)*vol(inp)*dtr*vold(inp)
            swt(inp)=den(inp)*vol(inp)*dtr*wold(inp)
          end do
          if (itidis.eq.2) then
            STOP 'caluvw; loldtime and uoldold not existent here'
          end if
        end if
      end if
    end if

    ! For u: sp => ap; for v: sp => spv; for w  sp => sp
#include "vdirnodep.h"
!$omp do schedule(static)
    do ivec = iis,iie
      inp=ijkvec(ivec)
      ap(inp)  = 0.d0
      spv(inp) = 0.d0
      sp(inp)  = 0.d0
    end do
!$omp end do
    
    ! Buoyancy term, density dependent on temperature or Boussinesq approximation
    if (lcal(ien)) then
      if (itbouy == 2) then
#include "vdirnodep.h"
!$omp do schedule(static)
        do ivec = iis,iie
          inp=ijkvec(ivec)
          heat=vol(inp)*(den(inp)-densit)
          su(inp)=su(inp) + gravx*heat
          sv(inp)=sv(inp) + gravy*heat
          sw(inp)=sw(inp) + gravz*heat
        end do
!$omp end do
      end if
      if (itbouy == 1) then
#include "vdirnodep.h"
!$omp do schedule(static)
        do ivec = iis,iie
          inp=ijkvec(ivec)
          heat=-fbeta(t(inp),nfr(m))*den(inp)* vol(inp)*(t(inp)-tf(nfr(m)))
          su(inp)=su(inp) + gravx*heat
          sv(inp)=sv(inp) + gravy*heat
          sw(inp)=sw(inp) + gravz*heat
        end do
!$omp end do
      end if
    end if
    
    ! Additional source terms for system with pressure gradient
    if (lpgrad) then
#include "vdirnodep.h"
!$omp do schedule(static)
      do ivec = iis,iie
        inp=ijkvec(ivec)
        su(inp)=su(inp)+pgradx*vol(inp)
        sv(inp)=sv(inp)+pgrady*vol(inp)
        sw(inp)=sw(inp)+pgradz*vol(inp)
      end do
!$omp end do nowait
    end if
!$omp end parallel

    ! Brinkman penalization
    if (lbrinkman) then
      do ivec = iis,iie
        inp=ijkvec(ivec)
        ap(inp)  = ap(inp)  + volFrac(inp)*vol(inp)
        spv(inp) = spv(inp) + volFrac(inp)*vol(inp)
        sp(inp)  = sp(inp)  + volFrac(inp)*vol(inp)
        su(inp)  = su(inp)  + volFrac(inp)*vol(inp)*usol(inp)
        sv(inp)  = sv(inp)  + volFrac(inp)*vol(inp)*vsol(inp)
        sw(inp)  = sw(inp)  + volFrac(inp)*vol(inp)*wsol(inp)
      end do
    end if

    ! Calculate terms integrated over surfaces (only inner surfaces) (east-north-top  cell - face)
    nie=nim-1
    nje=njm-1
    nke=nkm-1
    
    call celuvw(nie,njm,nkm,nj,1,nij,fx,fy,fz,ae,aw,f1,ngr,0,tayfx(1),tayfy(1),tayfz(1), &
#               include "listceluvw.h"
                ier)
    call celnbuvw(6,lbe,nyza,jks,nj,nk,                                            &
                  idfa(ngr,m,6,1)+idfa(ngr,m,6,4),idfa(ngr,m,6,2),idfa(ngr,m,6,3), &
                  nj,1,nij,fx,fy,fz,ae,f1,gds(iu,ngr),0,                           &
                  tayfx(1),tayfy(1),tayfz(1),                                      &
#                 include "listcelnbuvw.h"
                  ier)
    
    call celuvw(nim,nje,nkm,1,nij,nj,fy,fz,fx,an,as,f2,ngr,1,tayfy(nxyza+1),tayfz(nxyza+1),tayfx(nxyza+1), &
#               include "listceluvw.h"
                ier)
    call celnbuvw(5,lbn,nxza,iks,ni,nk,                                            &
                  idfa(ngr,m,5,1)+idfa(ngr,m,5,4),idfa(ngr,m,5,2),idfa(ngr,m,5,3), &
                  1,nij,nj,fy,fz,fx,an,f2,gds(iu,ngr),1,                           &
                  tayfy(nxyza+1),tayfz(nxyza+1),tayfx(nxyza+1),                    &
#                 include "listcelnbuvw.h"
                  ier)
    
    call celuvw(nim,njm,nke,nij,nj,1,fz,fx,fy,at,ab,f3,ngr,2,tayfz(2*nxyza+1),tayfx(2*nxyza+1),tayfy(2*nxyza+1), &
#               include "listceluvw.h"
                ier)
    call celnbuvw(4,lbt,nxya,ijs,ni,nj,                                            &
                  idfa(ngr,m,4,1)+idfa(ngr,m,4,4),idfa(ngr,m,4,2),idfa(ngr,m,4,3), &
                  nij,nj,1,fz,fx,fy,at,f3,gds(iu,ngr),2,                           &
                  tayfz(2*nxyza+1),tayfx(2*nxyza+1),tayfy(2*nxyza+1),              &
#                 include "listcelnbuvw.h"
                  ier)
    
    call celnbuvw(3,lbb,nxya, ijs,ni,nj,                              &
                  idfa(ngr,m,3,1),idfa(ngr,m,3,2),idfa(ngr,m,3,3),    &
                  nij,nj,1,fz,fx,fy,ab,f3,gds(iu,ngr),2,              &
                  tayfz(2*nxyza+1),tayfx(2*nxyza+1),tayfy(2*nxyza+1), &
#                 include "listcelnbuvw.h"
                  ier)
    
    call celnbuvw(2,lbs,nxza, iks,ni,nk,                           &
                  idfa(ngr,m,2,1),idfa(ngr,m,2,2),idfa(ngr,m,2,3), &
                  1,nij,nj,fy,fz,fx,as,f2,gds(iu,ngr),1,           &
                  tayfy(nxyza+1),tayfz(nxyza+1),tayfx(nxyza+1),    &
#                 include "listcelnbuvw.h"
                  ier)
    
    call celnbuvw(1,lbw,nyza,jks,nj,nk,                            &
                  idfa(ngr,m,1,1),idfa(ngr,m,1,2),idfa(ngr,m,1,3), &
                  nj,1,nij,fx,fy,fz,aw,f1,gds(iu,ngr),0,           &
                  tayfx(1),tayfy(1),tayfz(1),                      &
#                 include "listcelnbuvw.h"
                  ier)
  end if
end do

! Implement boundary conditions
call moduvw(ngr, &
#           include "listmoduvw.h"
            ier)

! Unsteady terms
if (loldfl) then
  do m=1,nblo
    if (lfluidr(nfr(m))) then
      call setind(ngr,m, &
#                 include "listsetind.h"
                  ier)
#include "vdirnodep.h"
!$omp parallel do schedule(static), default(shared),
!$omp&         private(ivec,inp,int,inb,inn,ins,ine,inw),
!$omp&         private(apall,aptu,aptv,aptw,sumu,sumv,sumw)
      do ivec = iis,iie
        inp=ijkvec(ivec)
        int=inp+nij
        inb=inp-nij
        inn=inp+1
        ins=inp-1
        ine=inp+nj
        inw=inp-nj
        apall= ae(inp)+aw(inp)+an(inp)+as(inp)+at(inp)+ab(inp)
        aptu = apall + ap(inp)
        aptv = apall + spv(inp)
        aptw = apall + sp(inp)
        sumu = at(inp)*u(int)+ab(inp)*u(inb)+an(inp)*u(inn)+as(inp)*u(ins)+ ae(inp)*u(ine)+aw(inp)*u(inw)
        sumv = at(inp)*v(int)+ab(inp)*v(inb)+an(inp)*v(inn)+as(inp)*v(ins)+ ae(inp)*v(ine)+aw(inp)*v(inw)
        sumw = at(inp)*w(int)+ab(inp)*w(inb)+an(inp)*w(inn)+as(inp)*w(ins)+ ae(inp)*w(ine)+aw(inp)*w(inw)
        sut(inp) = sut(inp) + dtidis * ( su(inp) + sumu - aptu * u(inp) )
        svt(inp) = svt(inp) + dtidis * ( sv(inp) + sumv - aptv * v(inp) )
        swt(inp) = swt(inp) + dtidis * ( sw(inp) + sumw - aptw * w(inp) )
        ap(inp)  = 0.d0
        spv(inp) = 0.d0
        sp(inp)  = 0.d0
      end do
!$omp end parallel do
    end if
  end do
  
  return
end if

! Modify coefficients for Crank-Nicolson
if (itidis == 3) then
  do m=1,nblo
    if (lfluidr(nfr(m))) then
      call setind(ngr,m, &
#                 include "listsetind.h"
                  ier)
#include "vdirnodep.h"
!$omp parallel do schedule(static), default(shared), private(ivec,inp)
      do ivec = iis,iie
        inp=ijkvec(ivec)
        ae(inp) = ae(inp) * ftidis
        aw(inp) = aw(inp) * ftidis
        as(inp) = as(inp) * ftidis
        an(inp) = an(inp) * ftidis
        at(inp) = at(inp) * ftidis
        ab(inp) = ab(inp) * ftidis
        ap(inp)  = ap(inp)  * ftidis
        spv(inp) = spv(inp) * ftidis
        sp(inp)  = sp(inp)  * ftidis
        su(inp)  = su(inp)  * ftidis
        sv(inp)  = sv(inp)  * ftidis
        sw(inp)  = sw(inp)  * ftidis
      end do
!$omp end parallel do
    end if
  end do
end if

! Add unsteady terms
if (lold) then
  do m=1,nblo
    if (lfluidr(nfr(m))) then
      call setind(ngr,m, &
#                 include "listsetind.h"
                  ier)
#include "vdirnodep.h"
!$omp parallel do schedule(static), default(shared), private(ivec,inp)
      do ivec = iis,iie
        inp=ijkvec(ivec)
        su(inp)=su(inp)+sut(inp)
        sv(inp)=sv(inp)+svt(inp)
        sw(inp)=sw(inp)+swt(inp)
      end do
!$omp end parallel do
    end if
  end do
end if

if (ltime) then
  do m=1,nblo
    if (lfluidr(nfr(m))) then
      call setind(ngr,m, &
#                 include "listsetind.h"
                  ier)
      
      if (lmvgr) then
#include "vdirnodep.h"
!$omp parallel do schedule(static), default(shared),
!$omp&            private(ivec,inp,apo)
        
        ! By theory this should be
        ! apo = (2*vol(inp) - volold(inp)) *den(inp) * dtr
        ! See prepstep.F for details.
        do ivec = iis,iie
          inp = ijkvec(ivec)
          apo = den(inp) * dtr * (volold(inp)+beta*(volold(inp)-0.5d0*vololdold(inp)))
          ap(inp) = ap(inp) + apo
          spv(inp)= spv(inp)+ apo
          sp(inp) = sp(inp) + apo
        end do
!$omp end parallel do
      else
#include "vdirnodep.h"
!$omp parallel do schedule(static), default(shared),
!$omp&            private(ivec,inp,apo)
        do ivec = iis,iie
          inp = ijkvec(ivec)
          apo = den(inp) * vol(inp) * dtr * (1.d0+0.5d0*beta)
          ap(inp) = ap(inp) + apo
          spv(inp)= spv(inp)+ apo
          sp(inp) = sp(inp) + apo
        end do
!$omp end parallel do
      end if
    end if
  end do
end if

if (.not.lsol) return

! Additional sources for FAS
if (lsor) then
  do m=1,nblo
    if (lfluidr(nfr(m))) then
      call setind(ngr,m, &
#                 include "listsetind.h"
                  ier)
#include "vdirnodep.h"
!$omp parallel do schedule(static), default(shared),
!$omp&         private(ivec,inp,int,inb,inn,ins,ine,inw),
!$omp&         private(apall,aptu,aptv,aptw,suc,svc,swc)
      do ivec = iis,iie
        inp=ijkvec(ivec)
        apall = ae(inp)+aw(inp)+an(inp)+as(inp)+at(inp)+ab(inp)
        aptu  = apall + ap(inp)
        aptv  = apall + spv(inp)
        aptw  = apall + sp(inp)
        int=inp+nij
        inb=inp-nij
        inn=inp+1
        ins=inp-1
        ine=inp+nj
        inw=inp-nj

        suc=at(inp)*u1(int)+ab(inp)*u1(inb)+an(inp)*u1(inn)+as(inp)*u1(ins)+ae(inp)*u1(ine)+aw(inp)*u1(inw)-aptu*u1(inp)
        svc=at(inp)*v1(int)+ab(inp)*v1(inb)+an(inp)*v1(inn)+as(inp)*v1(ins)+ae(inp)*v1(ine)+aw(inp)*v1(inw)-aptv*v1(inp)
        swc=at(inp)*w1(int)+ab(inp)*w1(inb)+an(inp)*w1(inn)+as(inp)*w1(ins)+ae(inp)*w1(ine)+aw(inp)*w1(inw)-aptw*w1(inp)

        resu(inp)=resu(inp)-suc-su(inp)
        resv(inp)=resv(inp)-svc-sv(inp)
        resw(inp)=resw(inp)-swc-sw(inp)
      end do
!$omp end parallel do
    end if
  end do
end if

call coefadd(ngr,urfru,urfmu,u,ap,su,apu,resu, &
#            include "listcoefadd.h"
             ier)
call sipsol(u,v,w,iu,ngr, &
#           include "listsipsol.h"
            ier)
call coefadd(ngr,urfrv,urfmv,v,spv,sv,apv,resv, &
#            include "listcoefadd.h"
             ier)
call sipsol(v,u,w,iv,ngr, &
#           include "listsipsol.h"
            ier)
call coefadd(ngr,urfrw,urfmw,w,sp,sw,apw,resw, &
#            include "listcoefadd.h"
             ier)
call sipsol(w,u,v,iw,ngr, &
#                 include "listsipsol.h"
                  ier)

end subroutine caluvw
