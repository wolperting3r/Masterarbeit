%% Pseudocode für Zach
% Codekommentare sind mittels  ! eingefügt
% Code für CDS Kruemmungsberechnung
% conc = concentration
% fx, fy, fz sind Zellgrößenverhältnisse in die jeweiligen Richtungen-- bei kartesischen Zellen (Quadrat/Würfel) ist fx = fy = fz = 0.5
! First gradients
! Initialize help coefficiebts be, bn, bt
be(1:ende)=0.d0
bn(1:ende)=0.d0
bt(1:ende)=0.d0
! Calculation of the concentration values on the cell faces 
! (saving in the help b-coefficients) 
do k = 1, (ende-1)
do i = 1, (ende-1)
do j = 1, (ende-1)
! Compute b-coef. at faces using cds
be(i, j, k)=conc(i, j, k)*(1.d0-fx(i, j, k))+conc(i+1, j, k)*fx(i, j, k)
bn(i, j, k)=conc(i, j, k)*(1.d0-fy(i, j, k))+conc(i, j+1, k)*fy(i, j, k)
bt(i, j, k)=conc(i, j, k)*(1.d0-fz(i, j, k))+conc(i, j, k+1)*fz(i, j, k)
end do
end do
end do
! Gradients calculation
do k = 2, (ende-1)
do i = 2, (ende-1)
do j = 2, (ende-1)
! Concentration differences between cell faces
cew = be(i, j, k)-be(i-1, j, k)  ! east-west
cns = bn(i, j, k)-bn(i, j-1, k)  ! north-south
ctb = bt(i, j, k)-bt(i, j, k-1)  ! top-bottom

! Concentration gradients
% ar1x, ar2x, ar3x bspw sind geometrieabhängige Vektoren, die aber im kartesischen Fall aber jeweils nur eine Komponente ungleich 0 haben
% in x ist nur ar1x ungleich 0 (=1), in y ist nur ar2y ungleich 0 (=1) usw
% vol ist einfach das Volumen der Zelle, da ar1x, ... eigentlich ein Vektor ist, dessen Länge der Fläche der Seite entspricht
cdx(i, j, k)=(ar1x(i, j, k)*cew+ar2x(i, j, k)*cns+ar3x(i, j, k)*ctb)/vol(i, j, k)
cdy(i, j, k)=(ar1y(i, j, k)*cew+ar2y(i, j, k)*cns+ar3y(i, j, k)*ctb)/vol(i, j, k)
cdz(i, j, k)=(ar1z(i, j, k)*cew+ar2z(i, j, k)*cns+ar3z(i, j, k)*ctb)/vol(i, j, k)
end do
end do
end do
! Second gradients for cdx
! Initialize help coefficiebts be, bn, bt
be(1:ende)=0.d0
bn(1:ende)=0.d0
bt(1:ende)=0.d0
! Calculation of the concentration values on the cell faces 
! (saving in the help b-coefficients) 
do k = 1, (ende-1)
do i = 1, (ende-1)
do j = 1, (ende-1)
! Compute b-coef. at faces using cds
be(i, j, k)=cdx(i, j, k)*(1.d0-fx(i, j, k))+cdx(i+1, j, k)*fx(i, j, k)
bn(i, j, k)=cdx(i, j, k)*(1.d0-fy(i, j, k))+cdx(i, j+1, k)*fy(i, j, k)
bt(i, j, k)=cdx(i, j, k)*(1.d0-fz(i, j, k))+cdx(i, j, k+1)*fz(i, j, k)
end do
end do
end do
! Gradients calculation
do k = 2, (ende-1)
do i = 2, (ende-1)
do j = 2, (ende-1)
! Concentration differences between cell faces
cew = be(i, j, k)-be(i-1, j, k)  ! east-west
cns = bn(i, j, k)-bn(i, j-1, k)  ! north-south
ctb = bt(i, j, k)-bt(i, j, k-1)  ! top-bottom

! Concentration gradients
% ar1x, ar2x, ar3x bspw sind geometrieabhängige Vektoren, die aber im kartesischen Fall aber jeweils nur eine Komponente ungleich 0 haben
% in x ist nur ar1x ungleich 0 (=1), in y ist nur ar2y ungleich 0 (=1) usw
% vol ist einfach das Volumen der Zelle, da ar1x, ... eigentlich ein Vektor ist, dessen Länge der Fläche der Seite entspricht
cdxx(i, j, k)=(ar1x(i, j, k)*cew+ar2x(i, j, k)*cns+ar3x(i, j, k)*ctb)/vol(i, j, k)
cdxy(i, j, k)=(ar1y(i, j, k)*cew+ar2y(i, j, k)*cns+ar3y(i, j, k)*ctb)/vol(i, j, k)
cdxz(i, j, k)=(ar1z(i, j, k)*cew+ar2z(i, j, k)*cns+ar3z(i, j, k)*ctb)/vol(i, j, k)
end do
end do
end do
! Second gradients for cdy
! Initialize help coefficiebts be, bn, bt
be(1:ende)=0.d0
bn(1:ende)=0.d0
bt(1:ende)=0.d0
! Calculation of the concentration values on the cell faces 
! (saving in the help b-coefficients) 
do k = 1, (ende-1)
do i = 1, (ende-1)
do j = 1, (ende-1)
! Compute b-coef. at faces using cds
be(i, j, k)=cdy(i, j, k)*(1.d0-fx(i, j, k))+cdy(i+1, j, k)*fx(i, j, k)
bn(i, j, k)=cdy(i, j, k)*(1.d0-fy(i, j, k))+cdy(i, j+1, k)*fy(i, j, k)
bt(i, j, k)=cdy(i, j, k)*(1.d0-fz(i, j, k))+cdy(i, j, k+1)*fz(i, j, k)
end do
end do
end do
! Gradients calculation
do k = 2, (ende-1)
do i = 2, (ende-1)
do j = 2, (ende-1)
! Concentration differences between cell faces
cew = be(i, j, k)-be(i-1, j, k)  ! east-west
cns = bn(i, j, k)-bn(i, j-1, k)  ! north-south
ctb = bt(i, j, k)-bt(i, j, k-1)  ! top-bottom

! Concentration gradients
% ar1x, ar2x, ar3x bspw sind geometrieabhängige Vektoren, die aber im kartesischen Fall aber jeweils nur eine Komponente ungleich 0 haben
% in x ist nur ar1x ungleich 0 (=1), in y ist nur ar2y ungleich 0 (=1) usw
% vol ist einfach das Volumen der Zelle, da ar1x, ... eigentlich ein Vektor ist, dessen Länge der Fläche der Seite entspricht
cdyx(i, j, k)=(ar1x(i, j, k)*cew+ar2x(i, j, k)*cns+ar3x(i, j, k)*ctb)/vol(i, j, k)
cdyy(i, j, k)=(ar1y(i, j, k)*cew+ar2y(i, j, k)*cns+ar3y(i, j, k)*ctb)/vol(i, j, k)
cdyz(i, j, k)=(ar1z(i, j, k)*cew+ar2z(i, j, k)*cns+ar3z(i, j, k)*ctb)/vol(i, j, k)
end do
end do
end do
! Second gradients for cdz
! Initialize help coefficiebts be, bn, bt
be(1:ende)=0.d0
bn(1:ende)=0.d0
bt(1:ende)=0.d0
! Calculation of the concentration values on the cell faces 
! (saving in the help b-coefficients) 
do k = 1, (ende-1)
do i = 1, (ende-1)
do j = 1, (ende-1)
! Compute b-coef. at faces using cds
be(i, j, k)=cdz(i, j, k)*(1.d0-fx(i, j, k))+cdz(i+1, j, k)*fx(i, j, k)
bn(i, j, k)=cdz(i, j, k)*(1.d0-fy(i, j, k))+cdz(i, j+1, k)*fy(i, j, k)
bt(i, j, k)=cdz(i, j, k)*(1.d0-fz(i, j, k))+cdz(i, j, k+1)*fz(i, j, k)
end do
end do
end do
! Gradients calculation
do k = 2, (ende-1)
do i = 2, (ende-1)
do j = 2, (ende-1)
! Concentration differences between cell faces
cew = be(i, j, k)-be(i-1, j, k)  ! east-west
cns = bn(i, j, k)-bn(i, j-1, k)  ! north-south
ctb = bt(i, j, k)-bt(i, j, k-1)  ! top-bottom

! Concentration gradients
% ar1x, ar2x, ar3x bspw sind geometrieabhängige Vektoren, die aber im kartesischen Fall aber jeweils nur eine Komponente ungleich 0 haben
% in x ist nur ar1x ungleich 0 (=1), in y ist nur ar2y ungleich 0 (=1) usw
% vol ist einfach das Volumen der Zelle, da ar1x, ... eigentlich ein Vektor ist, dessen Länge der Fläche der Seite entspricht
cdzx(i, j, k)=(ar1x(i, j, k)*cew+ar2x(i, j, k)*cns+ar3x(i, j, k)*ctb)/vol(i, j, k)
cdzy(i, j, k)=(ar1y(i, j, k)*cew+ar2y(i, j, k)*cns+ar3y(i, j, k)*ctb)/vol(i, j, k)
cdzz(i, j, k)=(ar1z(i, j, k)*cew+ar2z(i, j, k)*cns+ar3z(i, j, k)*ctb)/vol(i, j, k)
end do
end do
end do
! Krümmungsberechnung
% Im Code haben wir eine Bedingung ausser 0 < conc < 1, wann überhaupt eine Krümmung berechnet wird. Da musst du mal schauen, ob es bei dir Sinn macht.
xm = sqrt((x(i, j, k)-x(i-1, j, k))**2)
ym = sqrt((y(i, j, k)-y(i, j-1, k))**2)
zm = sqrt((z(i, j, k)-z(i, j, k-1))**2)
h = min(xm, ym, zm)
gh = 0.01/h
if (sqrt(cdx(inp)**2+cdy(inp)**2+cdz(inp)**2) > gh) then
        calccurv=.true.
end if
if (conc(i, j, k)<0.99d0 .and. conc(i, j, k)>0.01d0 .and. calccurv==.true.) then
        curvature(i, j, k)= -1 * (cdy(inp)**2*cdxx(inp)-2*cdx(inp)*cdy(inp)*cdxy(inp)+cdx(inp)**2*cdyy(inp)  + &
                cdx(inp)**2*cdzz(inp)-2*cdx(inp)*cdz(inp)*cdxz(inp)+cdz(inp)**2*cdxx(inp)  + &
                cdy(inp)**2*cdzz(inp)-2*cdy(inp)*cdz(inp)*cdyz(inp)+cdz(inp)**2*cdyy(inp)) / &
                (cdx(inp)**2+cdy(inp)**2+cdz(inp)**2)**(3.d0/2.d0)
end if
%
% Verbesserungen dieser einfachsten Form konnte ich feststellen, wenn ich die Konzentration für die Gradienten noch glätte, die Krümmung ebenfalls glätte und die geglättete Krümmung noch gewichte
%
% Die Glättung erfolgt über einen Chebyshevfilter. Für den folgenden Code habe ich Randbehandlungen rausgelassen. Eigentlich ist das Vorgehen vergleichsweise wie der Gaußfilter, nur ohne Diagonalnachbarn und
% mit anderen Koeffizienten. Es können dabei Werte über 1 oder unter 0 auftauchen, was aber nicht schlimm ist, da diese Werte nur für die Gradientenbestimmung und die Oberflächenspannungskraft verwendet werden, 
% jedoch nicht für den Transport der Konzentration.
% Die Konzentration habe ich mit dem LSP Degree 2 Koeffizienten und die Krümmung mit dem 5-Point Chebyshev gefiltert
% var = Variable wie conc oder curvature
select case (filter)
! LSP Degree 0
  case(1)
          ch1 = 1.d0
          ch2 = 1.d0
          ch3 = 1.d0
          ! Binominal 
  case (2)
          ch1 = 1d0
          ch2 = 4d0
          ch3 = 6d0
          ! 5-Point Chebyshev 
  case (3)
          ch1 = -1.d0
          ch2 = 4.d0
          ch3 = 10.d0
          ! LSP Degree 2
  case(4)
          ch1 = -3.d0
          ch2 = 12.d0
          ch3 = 17.d0
  end select
  ! denominator is sum (a*ch1+b*ch2+c*ch3+d*ch2+e*ch1/sum(ch1+ch2+ch3+ch2+ch1))
  ch4 = 2.d0*ch1+2.d0*ch2+1.d0*ch3
  do k = 3, (ende-2)
  do i = 3, (ende-2)
  do j = 3, (ende-2)
  ! i-direction
  help1(i, j, k)=(ch1*var(i-2, j, k)+ch2*var(i-1, j, k)+ch3*var(i, j, k)+ch2*var(i+1, j, k)+ch1*var(i+,j, k))/ch4
  ! j-direction
  help2(i, j, k)=(ch1*var(i, j-2, k)+ch2*var(i, j-1, k)+ch3*var(i, j, k)+ch2*var(i+1, j, k)+ch1*var(i+2, j, k))/ch4
  ! k-direction
  help3(i, j, k)=(ch1*var(i, j, k-2)+ch2*var(i, j, k-1)+ch3*var(i, j, k)+ch2*var(i, j, k+1)+ch1*var(i, j, k+2))/ch4
  ! final value
  help4(i, j, k)=1.d0/3.d0*(help1(i, j, k)+help2(i, j, k)+help3(i, j, k))
  end do  ! j
  end do  ! i
  end do  ! k
  % Überschreiben der Eingangsvariable
  do k = 3, (ende-2)
  do i = 3, (ende-2)
  do j = 3, (ende-2)
  var(i, j, k)=help4(i, j, k)
  end do  ! j
  end do  ! i
  end do  ! k    
  % Über die Gewichtung haben wir gesprochen. Die Krümmung wird dort auch angewendet, wo vorher auch eine Krümmung berechnet wurde, damit die nicht noch ausgeschmiert wird.
  % weightsexp ist bei mir immer = 1
  ! Initialize vectors
  weights(1:ende)=0d0
  tempcurv(1:ende)=0d0
  ! Calculate weights
  do k = 1, ende
  do i = 1, ende
  do j = 1, ende
  ! more weight to cells with 0.5 concentration, less close to 0 or 1
  weights(i, j, k)=(1-2d0*abs(0.5d0-conc(i, j, k)))**(weightsexp)
  end do
  end do
  end do
  ! Calculate new curvature with weights
  do k = 2, nkm
  do i = 2, nim
  do j = 2, njm
  if (dscurv2(i, j, k)/=0d0) then
          tempcurv(inp)=(curvature(i, j, k)*weights(i, j, k) + &
                  curvature(i+1, j, k)*weights(i+1, j, k)+ &
                  curvature(i-1, j, k)*weights(i-1, j, k)+ &
                  curvature(i, j+1, k)*weights(i, j+1, k)+ &
                  curvature(i, j-1, k)*weights(i, j-1, k)+ &
                  curvature(i, j, k+1)*weights(i, j, k+1)+ &
                  curvature(i, j, k-1)*weights(i, j, k-1))/ &
                  (weights(i, j, k)+weights(i+1, j, k)+weights(i-1, j, k)+weights(i, j+1, k)+weights(i, j-1, k)+weights(i, j, k+1)+weights(i, j, k-1))
  end if
  end do
  end do
  end do
  % Überschreiben der alten Krümmungen
  do k = 2, nkm
  do i = 2, nim
  do j = 2, njm
  inp = ha(i, j, k)
  if (dscurv2(inp)/=0d0) then
          dscurv2(inp)=tempcurv(inp)
  end if
  end do
  end do
  end do
