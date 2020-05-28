%% Pseudocode für Zach
% Codekommentare sind mittels  ! eingefügt
% Verschmieren mittels Gaußverteilung (in 2D-- k Index spielt keine Rolle)
% conc = concentration
% tempconc = temporäre concentration
do k = 1, ende
do i = 1, ende
do j = 1, ende
tempconc = 4*conc(i, j, k)
tempconc = tempconc+2*conc(i-1, j, k)+2*conc(i, j-1, k) + 2*conc(i+1, j, k) + 2*conc(i, j+1, k)
tempconc = tempconc+conc(i-1, j-1, k) + conc(i+1, j+1, k) + conc(i-1, j+1, k) + conc(i+1, j-1, k)
icount = 16
conc(i, j, k)=tempconc/icount           
end do
end do
end do
% Verschmieren mittels unserer eigenen Routine
% fx, fy, fz sind Zellgrößenverhältnisse in die jeweiligen Richtungen-- bei kartesischen Zellen (Quadrat/Würfel) ist fx = fy = fz = 0.5
! Initialization of vectors
be(1:ende)=0.d0
bn(1:ende)=0.d0
bt(1:ende)=0.d0
bp(1:ende)=0.d0
bm(1:ende)=0.d0
! Interpolation on the CV's faces  
! (saving in the help b-coefficients) 
do k = 1, (ende-1)
do i = 1, (ende-1)
do j = 1, (ende-1)
! Linear interpolation
be(i, j, k)=b(i, j, k)*(1.d0-fx(i, j, k))+b(i+1, j, k)*fx(inp)  ! 1/2 b_i+1/2 b_i+1
bn(i, j, k)=b(i, j, k)*(1.d0-fy(i, j, k))+b(i, j+1, k)*fy(inp)  ! 1/2 b_i+1/2 b_i+1
bt(i, j, k)=b(i, j, k)*(1.d0-fz(i, j, k))+b(i, j, k+1)*fz(inp)  ! 1/2 b_i+1/2 b_i+1
end do
end do
end do
! Calculate weighting in inner area
do k = 2, (ende-1)
do i = 2, (ende-1)
do j = 2, (ende-1)
% Habe hier einen Teil rausgelöscht, der SE, SN, ST in Abhängigkeit der Zellgeometrie berechnen kann, aber wir verwenden immer (unabhägig von Zellgeometrie) folgendes:
SE = 1.d0
SN = 1.d0
ST = 1.d0
bm(i, j, k)   = bm(i, j, k)+be(inp)*SE+bn(i, j, k)*SN+bt(i, j, k)*ST
bm(i+1, j, k) = bm(i+1, j, k)+be(inp)*SE
bm(i, j+1, k) = bm(i, j+1, k)+bn(inp)*SN
bm(i, j, k+1) = bm(i, j, k+1)+bt(inp)*ST
bp(i, j, k)=bp(i, j, k)+SE+SN+ST
bp(i+1, j, k)=bp(i+1, j, k)+SE
bp(i, j+1, k))=bp(i, j+1, k))+SN
bp(i, j, k+1)=bp(i, j, k+1)+ST
end do
end do
end do
% Hier sind im Originalcode noch die Sonderbehandlungen für die Randgebiete drin, die aber eig auch nicht wichtig sind für deine Anwendung im Moment
! Normalize
do k = 2, (ende-1)
do i = 2, (ende-1)
do j = 2, (ende-1)
bm(i, j, k)=bm(i, j, k)/(bp(i, j, k)+small)
end do
end do
end do
