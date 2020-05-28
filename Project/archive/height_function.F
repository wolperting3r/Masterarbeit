%% Pseudocode für Zach
% Codekommentare sind mittels  ! eingefügt
% Code für HF Kruemmungsberechnung (einfachste Variante)
% conc = concentration

% Zuerst werden die Gradienten bestimmt über die gleiche Funktionen wie im CDS Fall--> cdx, cdy, cdz vorhanden für alle CV (Kontrollvolumen) 

do k = 2, (ende-1)
do i = 2, (ende-1)
do j = 2, (ende-1)
hi(1)=0.d0 
hi(2)=0.d0
hi(3)=0.d0
% Ich weiß gerade nicht mehr, ob hier oder später bei der Krümmungsberechnung nicht auch die andere Bedingung besser ist (sqrt(cdx(i, j, k)**2+cdy(i, j, k)**2+cdz(i, j, k)**2) > gh).
% Falls es so nicht gut funktioniert, kannst du mal kurz schauen, ob es durch die Bedingung besser wird. Wobei ich denke, dass diese Bedingungen im Moment bei dir Relevanz haben.
! Interface
if ((conc(i, j, k) > 0.01d0) .and. (c(i, j, k) < 0.99d0)) then

        ! n_y  > n_x => 7x3 Matrix
        if (abs(cdy(i, j, k)) > abs(cdx(i, j, k))) then
                ! only equidistant
                delx = x(i, j, k)-x(i-1, j, k) 
                dely = y(i, j, k)-y(i, j-1, k)  
                ! central line
                hi(1)=hi(1)+conc(i-1, j, k)*dely
                hi(2)=hi(2)+conc(i, j, k)  *dely
                hi(3)=hi(3)+conc(i+1, j, k)*dely
                ! neighbor lines
                do count = 1, 3       ! 6 x in y-direction
                hi(1)=hi(1)+(conc(i-1, j-count, k) +conc(i-1, j+count, k)) *dely  ! left
                hi(2)=hi(2)+(conc(i,  j-count, k) +conc(i  ,j+count, k)) *dely  ! middle
                hi(3)=hi(3)+(conc(i+1, j-count, k) +conc(i+1, j+count, k)) *dely  ! right
                end do
                ! First derivatives
                lstx1=(hi(3)-hi(1))/(2.d0*delx)
                lstx2=(hi(3)-2.d0*hi(2)+hi(1))/(delx**2.d0)
                ! nx > ... =>3x7 Matrix
        else if (abs(cdx(i, j, k)) > abs(cdy(i, j, k))) then 
                ! only equidistant
                delx = x(i, j, k)-x(i-1, j, k) 
                dely = y(i, j, k)-y(i, j-1, k)  
                ! central line
                hi(1)=hi(1)+conc(i, j-1, k)*delx
                hi(2)=hi(2)+conc(i, j, k)  *delx
                hi(3)=hi(3)+conc(i, j+1, k)*delx
                ! neighbor lines
                do count = 1, 3       ! 6 x in x-direction
                hi(1)=hi(1)+(conc(i-count, j-1, k) +conc(i+count, j-1, k)) *delx  ! top
                hi(2)=hi(2)+(conc(i-count,  j, k) +conc(i+count,  j, k)) *delx  ! middle
                hi(3)=hi(3)+(conc(i-count, j+1, k) +conc(i+count, j+1, k)) *delx  ! bottom
                end do
                ! First derivatives
                lstx1=(hi(3)-hi(1))/(2.d0*dely)
                lstx2=(hi(3)-2.d0*hi(2)+hi(1))/(dely**2.d0)
        end if
        ! Curvature
        curvature(i, j, k)=lstx2/(1+lstx1**2.d0)**(d3/d2)

end if 
do k = 2, (ende-1)
do i = 2, (ende-1)
do j = 2, (ende-1)  
