%===============================================================
% VOF-ML.m
% A very simple Navier-Stokes solver for a drop falling in a
% rectangular box, using a conservative form of the equations. 
% A 3-order explicit projection method and centered in space 
% discretizationa are used. The density is advected by a front 
% tracking scheme and surface tension and variable viscosity is
% included. The VOF volume fraction is found directly from the
% front and the surface tension is found by machine learning
% Author: Yinghe Qi and Gretar Tryggvason (5/25/2018)
%===============================================================
clear
close all

Lx=1.0;Ly=1.0;gx=0.0;gy=0.0; sigma=10; % Domain size and
rho1=1.0; rho2=2.0; m1=0.01; m2=0.02;     % physical variables
unorth=0; usouth=0; veast=0; vwest=0; time=0.0; 
rad=0.15; xc=0.5; yc=0.7;          % Initial drop size and location

%-------------------- Numerical variables ----------------------
nx=128;ny=128;dt=0.0005;nstep=100; maxit=200;maxError=0.001;beta=1.5; Nf=1200;

%-------------------- Zero various arrys -----------------------
u=zeros(nx+1,ny+2);  v=zeros(nx+2,ny+1);  p=zeros(nx+2,ny+2);
ut=zeros(nx+1,ny+2); vt=zeros(nx+2,ny+1); tmp1=zeros(nx+2,ny+2); 
uu=zeros(nx+1,ny+1); vv=zeros(nx+1,ny+1); tmp2=zeros(nx+2,ny+2);
fx=zeros(nx+2,ny+2); fy=zeros(nx+2,ny+2); r=zeros(nx+2,ny+2);
r=zeros(nx+2,ny+2);  chi=zeros(nx+2,ny+2); 
m=zeros(nx+2,ny+2);  d=zeros(nx+2,ny+2); 
xf=zeros(1,Nf+2); yf=zeros(1,Nf+2); 
uf=zeros(1,Nf+2); vf=zeros(1,Nf+2);
tx=zeros(1,Nf+2); ty=zeros(1,Nf+2);
un=zeros(nx+1,ny+2); vn=zeros(nx+2,ny+1);   % Used for 
rn=zeros(nx+2,ny+2); mn=zeros(nx+2,ny+2);   % higher order
xfn=zeros(1,Nf+2); yfn=zeros(1,Nf+2);       % in time

dx=Lx/nx;dy=Ly/ny;                          % Set the grid 
for i=1:nx+2; x(i)=dx*(i-1.5);end; for j=1:ny+2; y(j)=dy*(j-1.5);end;

%-------------------- Initial Conditions -----------------------
r=zeros(nx+2,ny+2)+rho1;m=zeros(nx+2,ny+2)+m1; % Set density and viscosity
x0=0.5;y0=0.5;r0=0.15;n=3;amp=0.05; DTheta=2*pi/(Nf);
for l=1:Nf+2
 theta=DTheta*(l-1);
 rad(l)=r0+amp*cos(n*theta);
 xf(l)=x0+rad(l)*cos(theta); yf(l)=y0+rad(l)*sin(theta); end                                         
% ---------------- construct the marker --------------------- 

chi=zeros(nx+2,ny+2);    % Set the initial value of chi

for i=1:Nf
  ip1=floor(xf(i)/dx)+2;ip2=floor(xf(i+1)/dx)+2;
  jp1=floor(yf(i)/dy)+2;jp2=floor(yf(i+1)/dy)+2;

  xs(1)=xf(i);xs(2)=xf(i+1);xs(3)=xf(i+1);xs(4)=xf(i+1);  % Add two in-
  ys(1)=yf(i);ys(2)=yf(i+1);ys(3)=yf(i+1);ys(4)=yf(i+1);  % between points

  if (ip1 ~= ip2)
    if (ip1 < ip2) xv=(ip2-2)*dx; end; if (ip1 > ip2) xv=(ip1-2)*dx; end;
    yv=yf(i)+((yf(i+1)-yf(i))/(xf(i+1)-xf(i)))*(xv-xf(i));
  end; 
  
  if (jp1 ~= jp2)
    if (jp1 < jp2) yh=(jp2-2)*dy; end; if (jp1 > jp2) yh=(jp1-2)*dy; end;
    xh=xf(i)+((xf(i+1)-xf(i))/(yf(i+1)-yf(i)))*(yh-yf(i));
  end;

  if (ip1 ~= ip2) & (jp1 == jp2), xs(2)=xv;ys(2)=yv; end
  if (jp1 ~= jp2) & (ip1 == ip2) xs(2)=xh;ys(2)=yh; end
   
  if (ip1 < ip2) & (jp1 ~= jp2),
      xs(2)=xv;ys(2)=yv; xs(3)=xh;ys(3)=yh;
      if(xv > xh) xs(2)=xh;ys(2)=yh; xs(3)=xv;ys(3)=yv;end
  end    
  if (ip1 > ip2) & (jp1 ~= jp2),
      xs(2)=xv;ys(2)=yv; xs(3)=xh;ys(3)=yh;
      if(xv < xh) xs(2)=xh;ys(2)=yh; xs(3)=xv;ys(3)=yv;end
  end    

   for j=1:3        
     ip=floor(0.5*(xs(j)+xs(j+1))/dx)+2;
     jp=floor(0.5*(ys(j)+ys(j+1))/dy)+2;
     ddx=-(xs(j+1)-xs(j));
     chi(ip,jp)=chi(ip,jp)+(0.5*(ys(j)+ys(j+1))-dy*(jp-2))*ddx/dx/dy;
     for k=1:jp-1;chi(ip,k)=chi(ip,k)+dy*ddx/dx/dy; end     % start at the bottom
   end
end

% ----- end of constructing the marker ------------------- 
hold off,contour(x(2:nx+1),y(2:ny+1),chi(2:nx+1,2:nx+1)'),axis equal,axis([0 Lx 0 Ly]);
hold on;plot(xf(1:Nf),yf(1:Nf),'k','linewidth',3);pause(0.01)               

%---------------------- START TIME LOOP ------------------------
for is=1:nstep,is
  un=u; vn=v; rn=r; mn=m; xfn=xf; yfn=yf;  % Higher order
  for substep=1:3                          % in time

%---------------------- Advect the Front -----------------------
	for l=2:Nf+1                       % Interpolate the Front Velocities
      ip=floor(xf(l)/dx)+1; jp=floor((yf(l)+0.5*dy)/dy)+1;
      ax=xf(l)/dx-ip+1;ay=(yf(l)+0.5*dy)/dy-jp+1;	   
      uf(l)=(1.0-ax)*(1.0-ay)*u(ip,jp)+ax*(1.0-ay)*u(ip+1,jp)+...
		             (1.0-ax)*ay*u(ip,jp+1)+ax*ay*u(ip+1,jp+1);
						
      ip=floor((xf(l)+0.5*dx)/dx)+1; jp=floor(yf(l)/dy)+1;
      ax=(xf(l)+0.5*dx)/dx-ip+1;ay=yf(l)/dy-jp+1;
	  vf(l)=(1.0-ax)*(1.0-ay)*v(ip,jp)+ax*(1.0-ay)*v(ip+1,jp)+...
		             (1.0-ax)*ay*v(ip,jp+1)+ax*ay*v(ip+1,jp+1);
    end     

    for l=2:Nf+1, xf(l)=xf(l)+dt*uf(l); yf(l)=yf(l)+dt*vf(l);end % Move the
	xf(1)=xf(Nf+1);yf(1)=yf(Nf+1);xf(Nf+2)=xf(2);yf(Nf+2)=yf(2); % Front
 	
%-------------- Update the marker function ---------------------
% ---------------- construct the marker --------------------- 

chi=zeros(nx+2,ny+2);    % Set the initial value of chi

for i=1:Nf
  ip1=floor(xf(i)/dx)+2;ip2=floor(xf(i+1)/dx)+2;
  jp1=floor(yf(i)/dy)+2;jp2=floor(yf(i+1)/dy)+2;

  xs(1)=xf(i);xs(2)=xf(i+1);xs(3)=xf(i+1);xs(4)=xf(i+1);  % Add two in-
  ys(1)=yf(i);ys(2)=yf(i+1);ys(3)=yf(i+1);ys(4)=yf(i+1);  % between points

  if (ip1 ~= ip2)
    if (ip1 < ip2) xv=(ip2-2)*dx; end; if (ip1 > ip2) xv=(ip1-2)*dx; end;
    yv=yf(i)+((yf(i+1)-yf(i))/(xf(i+1)-xf(i)))*(xv-xf(i));
  end; 
  
  if (jp1 ~= jp2)
    if (jp1 < jp2) yh=(jp2-2)*dy; end; if (jp1 > jp2) yh=(jp1-2)*dy; end;
    xh=xf(i)+((xf(i+1)-xf(i))/(yf(i+1)-yf(i)))*(yh-yf(i));
  end;

  if (ip1 ~= ip2) & (jp1 == jp2), xs(2)=xv;ys(2)=yv; end
  if (jp1 ~= jp2) & (ip1 == ip2) xs(2)=xh;ys(2)=yh; end
   
  if (ip1 < ip2) & (jp1 ~= jp2),
      xs(2)=xv;ys(2)=yv; xs(3)=xh;ys(3)=yh;
      if(xv > xh) xs(2)=xh;ys(2)=yh; xs(3)=xv;ys(3)=yv;end
  end    
  if (ip1 > ip2) & (jp1 ~= jp2),
      xs(2)=xv;ys(2)=yv; xs(3)=xh;ys(3)=yh;
      if(xv < xh) xs(2)=xh;ys(2)=yh; xs(3)=xv;ys(3)=yv;end
  end    

   for j=1:3        
     ip=floor(0.5*(xs(j)+xs(j+1))/dx)+2;
     jp=floor(0.5*(ys(j)+ys(j+1))/dy)+2;
     ddx=-(xs(j+1)-xs(j));
     chi(ip,jp)=chi(ip,jp)+(0.5*(ys(j)+ys(j+1))-dy*(jp-2))*ddx/dx/dy;
     for k=1:jp-1;chi(ip,k)=chi(ip,k)+dy*ddx/dx/dy; end     % start at the bottom
   end
end

% ----- end of constructing the marker -------------------        
%-------------------- Update the density ----------------------
    ro=r;
    for i=1:nx+2,for j=1:ny+2
      r(i,j)=rho1+(rho2-rho1)*chi(i,j);
    end,end 

%----------------- find the surface tension using NN ------------
cur=zeros(nx+2,ny+2);
norm=zeros(nx+2,ny+2);
normx=zeros(nx+2,ny+2);
normy=zeros(nx+2,ny+2);
length=zeros(nx+2,ny+2);
ip1=-1;
jp1=-1;
for l=1:Nf
    ip=floor(xf(l)/dx)+2;
    jp=floor(yf(l)/dy)+2;
    if (ip~=ip1)||(jp~=jp1)
        ip1=ip;
        jp1=jp;
        cur(ip1,jp1)=NNCircle2([chi(ip1-1,jp1-1),chi(ip1-1,jp1),chi(ip1-1,jp1+1), ...
            chi(ip1,jp1-1),chi(ip1,jp1),chi(ip1,jp1+1),chi(ip1+1,jp1-1), ...
            chi(ip1+1,jp1),chi(ip1+1,jp1+1)]')/dx; % find the curvature
        normx(ip1,jp1)=(chi(ip1+1,jp1)-chi(ip1-1,jp1))/2/dx; % find the norm
        normy(ip1,jp1)=(chi(ip1,jp1+1)-chi(ip1,jp1-1))/2/dy;
        norm(ip1,jp1)=sqrt(normx(ip1,jp1)^2+normy(ip1,jp1)^2);
        normx(ip1,jp1)=normx(ip1,jp1)/norm(ip1,jp1);
        normy(ip1,jp1)=normy(ip1,jp1)/norm(ip1,jp1);
        i=sqrt(-1);
        theta1=angle(normx(ip1,jp1)+i*normy(ip1,jp1));
        theta1=min(abs([theta1,theta1-pi/2,theta1-pi,theta1+pi/2,theta1+pi]));
        if chi(ip1,jp1)<(tan(theta1)/2)
            length(ip1,jp1)=sqrt(2*chi(ip1,jp1)/tan(theta1))/cos(theta1)*dx;
        elseif chi(ip1,jp1)<(1-tan(theta1)/2)
            length(ip1,jp1)=1/cos(theta1)*dx;
        else
            length(ip1,jp1)=sqrt(2*(1-chi(ip1,jp1))/tan(theta1))/cos(theta1)*dx;
        end
    end
end

%----------------- distribute the surface tension --------------
fx=zeros(nx+1,ny+2);
fy=zeros(nx+2,ny+1);
for i=2:nx % x component of surface tension
    for j=2:ny+1
        fx(i,j)=0.5*sigma*((cur(i,j))*normx(i,j)*length(i,j) ...
            +(cur(i+1,j))*normx(i+1,j)*length(i+1,j))/dx/dy;
    end
end
for i=2:nx+1
    for j=2:ny
        fy(i,j)=0.5*sigma*((cur(i,j))*normy(i,j)*length(i,j) ...
            +(cur(i,j+1))*normy(i,j+1)*length(i,j+1))/dx/dy;
    end
end
fx(1:nx+1,2)=fx(1:nx+1,2)+fx(1:nx+1,1);           % Correct boundary
fx(1:nx+1,ny+1)=fx(1:nx+1,ny+1)+fx(1:nx+1,ny+2);  % values for the
fy(2,1:ny+1)=fy(2,1:ny+1)+fy(1,1:ny+1);           % surface force
fy(nx+1,1:ny+1)=fy(nx+1,1:ny+1)+fy(nx+2,1:ny+1);  % on the grid

    
%------------- Set tangential velocity at boundaries -----------	     
    u(1:nx+1,1)=2*usouth-u(1:nx+1,2);u(1:nx+1,ny+2)=2*unorth-u(1:nx+1,ny+1);
    v(1,1:ny+1)=2*vwest-v(2,1:ny+1);v(nx+2,1:ny+1)=2*veast-v(nx+1,1:ny+1);

%-------------- Find the predicted velocities ------------------	     
    for i=2:nx,for j=2:ny+1      % Temporary u-velocity-advection
      ut(i,j)=(2.0/(r(i+1,j)+r(i,j)))*(0.5*(ro(i+1,j)+ro(i,j))*u(i,j)+ dt*( ...
      -(0.25/dx)*(ro(i+1,j)*(u(i+1,j)+u(i,j))^2-ro(i,j)*(u(i,j)+u(i-1,j))^2)...
      -(0.0625/dy)*( (ro(i,j)+ro(i+1,j)+ro(i,j+1)+ro(i+1,j+1))*             ...
                                       (u(i,j+1)+u(i,j))*(v(i+1,j)+v(i,j))  ...
      -(ro(i,j)+ro(i+1,j)+ro(i+1,j-1)+ro(i,j-1))*(u(i,j)                    ...
                                       +u(i,j-1))*(v(i+1,j-1)+v(i,j-1)))    ...
                                  + 0.5*(ro(i+1,j)+ro(i,j))*gx              ...
                                  +fx(i,j)*(ro(i+1,j)+ro(i,j))/2/1.5));
    end,end

    for i=2:nx+1,for j=2:ny       % Temporary v-velocity-advection 
      vt(i,j)=(2.0/(r(i,j+1)+r(i,j)))*(0.5*(ro(i,j+1)+ro(i,j))*v(i,j)+ dt*( ...     
      -(0.0625/dx)*( (ro(i,j)+ro(i+1,j)+ro(i+1,j+1)+ro(i,j+1))*             ...
                                        (u(i,j)+u(i,j+1))*(v(i,j)+v(i+1,j)) ...
                  - (ro(i,j)+ro(i,j+1)+ro(i-1,j+1)+ro(i-1,j))*              ...
                                    (u(i-1,j+1)+u(i-1,j))*(v(i,j)+v(i-1,j)))...                                 
      -(0.25/dy)*(ro(i,j+1)*(v(i,j+1)+v(i,j))^2-ro(i,j)*(v(i,j)+v(i,j-1))^2)...
                                  + 0.5*(ro(i,j+1)+ro(i,j))*gy              ...
                                  +fy(i,j)*(ro(i,j)+ro(i,j+1))/2/1.5));    
    end,end
        
    for i=2:nx,for j=2:ny+1      % Temporary u-velocity-viscosity
      ut(i,j)=ut(i,j)+(2.0/(r(i+1,j)+r(i,j)))*dt*(...                                         
               +(1./dx)*2.*(m(i+1,j)*(1./dx)*(u(i+1,j)-u(i,j)) -       ...
                  m(i,j)  *(1./dx)*(u(i,j)-u(i-1,j)) )                 ...
         +(1./dy)*( 0.25*(m(i,j)+m(i+1,j)+m(i+1,j+1)+m(i,j+1))*        ...
           ((1./dy)*(u(i,j+1)-u(i,j)) + (1./dx)*(v(i+1,j)-v(i,j)) ) -  ...
                0.25*(m(i,j)+m(i+1,j)+m(i+1,j-1)+m(i,j-1))*            ...
          ((1./dy)*(u(i,j)-u(i,j-1))+ (1./dx)*(v(i+1,j-1)- v(i,j-1))) ) ) ;
    end,end
       
    for i=2:nx+1,for j=2:ny       % Temporary v-velocity-viscosity 
          vt(i,j)=vt(i,j)+(2.0/(r(i,j+1)+r(i,j)))*dt*(...
            +(1./dx)*( 0.25*(m(i,j)+m(i+1,j)+m(i+1,j+1)+m(i,j+1))*     ...
           ((1./dy)*(u(i,j+1)-u(i,j)) + (1./dx)*(v(i+1,j)-v(i,j)) ) -  ...
                0.25*(m(i,j)+m(i,j+1)+m(i-1,j+1)+m(i-1,j))*            ...
          ((1./dy)*(u(i-1,j+1)-u(i-1,j))+ (1./dx)*(v(i,j)- v(i-1,j))) )...
           +(1./dy)*2.*(m(i,j+1)*(1./dy)*(v(i,j+1)-v(i,j)) -           ...
                  m(i,j) *(1./dy)*(v(i,j)-v(i,j-1)) ) ) ;    
    end,end   

%------------------ Solve the Pressure Equation ----------------    
    rt=r; lrg=1000;   % Compute source term and the coefficient for p(i,j)
    rt(1:nx+2,1)=lrg;rt(1:nx+2,ny+2)=lrg;
    rt(1,1:ny+2)=lrg;rt(nx+2,1:ny+2)=lrg;

    for i=2:nx+1,for j=2:ny+1
      tmp1(i,j)= (0.5/dt)*( (ut(i,j)-ut(i-1,j))/dx+(vt(i,j)-vt(i,j-1))/dy );
      tmp2(i,j)=1.0/( (1./dx)*(1./(dx*(rt(i+1,j)+rt(i,j)))+   ...
                               1./(dx*(rt(i-1,j)+rt(i,j))) )+ ...
                      (1./dy)*(1./(dy*(rt(i,j+1)+rt(i,j)))+   ...
                               1./(dy*(rt(i,j-1)+rt(i,j))) )   );
    end,end

    for it=1:maxit	               % Solve for pressure by SOR
      oldArray=p;
      for i=2:nx+1,for j=2:ny+1
        p(i,j)=(1.0-beta)*p(i,j)+beta* tmp2(i,j)*(        ...
        (1./dx)*( p(i+1,j)/(dx*(rt(i+1,j)+rt(i,j)))+      ...
                  p(i-1,j)/(dx*(rt(i-1,j)+rt(i,j))) )+ ...
        (1./dy)*( p(i,j+1)/(dy*(rt(i,j+1)+rt(i,j)))+      ...
                  p(i,j-1)/(dy*(rt(i,j-1)+rt(i,j))) ) - tmp1(i,j));
      end,end
      if max(max(abs(oldArray-p))) <maxError, break, end
    end
                                      
    for i=2:nx,for j=2:ny+1   % Correct the u-velocity 
      u(i,j)=ut(i,j)-dt*(2.0/dx)*(p(i+1,j)-p(i,j))/(r(i+1,j)+r(i,j));
    end,end
      
    for i=2:nx+1,for j=2:ny   % Correct the v-velocity
      v(i,j)=vt(i,j)-dt*(2.0/dy)*(p(i,j+1)-p(i,j))/(r(i,j+1)+r(i,j));
    end,end

    for i=1:nx+2,for j=1:ny+2 % Update the viscosity
      m(i,j)=m1+(m2-m1)*chi(i,j);
    end,end 

    if substep==2, % Higher order (RK-3) in time
      u=0.75*un+0.25*u; v=0.75*vn+0.25*v; r=0.75*rn+0.25*r;
      m=0.75*mn+0.25*m; xf=0.75*xfn+0.25*xf; yf=0.75*yfn+0.25*yf;
    elseif substep==3
      u=(1/3)*un+(2/3)*u; v=(1/3)*vn+(2/3)*v; r=(1/3)*rn+(2/3)*r;
      m=(1/3)*mn+(2/3)*m; xf=(1/3)*xfn+(2/3)*xf; yf=(1/3)*yfn+(2/3)*yf;
    end
    
  end               % End of sub-iteration for RK-3 time integration

%--------------- Add and deleate points in the Front -----------
  xfold=xf;yfold=yf; l1=1;
  for l=2:Nf+1
    ds=sqrt( ((xfold(l)-xf(l1))/dx)^2 + ((yfold(l)-yf(l1))/dy)^2);
    if (ds > 0.5)
      l1=l1+1;xf(l1)=0.5*(xfold(l)+xf(l1-1));yf(l1)=0.5*(yfold(l)+yf(l1-1));
      l1=l1+1;xf(l1)=xfold(l);yf(l1)=yfold(l);
    elseif (ds < 0.25)
       % DO NOTHING!
    else
      l1=l1+1;xf(l1)=xfold(l);yf(l1)=yfold(l);
    end    
  end
  Nf=l1-1;
  xf(1)=xf(Nf+1);yf(1)=yf(Nf+1);xf(Nf+2)=xf(2);yf(Nf+2)=yf(2);
  % Smooth the front
  xfold=xf; yfold=yf; for l=2:Nf+1; w=0.7;
      xf(l)=w*xfold(l)+0.5*(1-w)*(xfold(l+1)+xfold(l-1));
      yf(l)=w*yfold(l)+0.5*(1-w)*(yfold(l+1)+yfold(l-1)); end
  xf(1)=xf(Nf+1); yf(1)=yf(Nf+1); xf(Nf+2)=xf(2); yf(Nf+2)=yf(2);
  
%----------------- Compute Diagnostic quantitites --------------
  Area(is)=0; CentroidX(is)=0; CentroidY(is)=0; Time(is)=time;
  Perim(is)=0.0;
  for l=2:Nf+1, Area(is)=Area(is)+...
      0.25*((xf(l+1)+xf(l))*(yf(l+1)-yf(l))-(yf(l+1)+yf(l))*(xf(l+1)-xf(l)));  
    CentroidX(is)=CentroidX(is)+...
      0.125*((xf(l+1)+xf(l))^2+(yf(l+1)+yf(l))^2)*(yf(l+1)-yf(l));
    CentroidY(is)=CentroidY(is)-...
      0.125*((xf(l+1)+xf(l))^2+(yf(l+1)+yf(l))^2)*(xf(l+1)-xf(l));
    dss=sqrt((xf(l+1)-xf(l))^2 + (yf(l+1)-yf(l))^2); Perim(is)=Perim(is)+dss;
  end
  CentroidX(is)=CentroidX(is)/Area(is);CentroidY(is)=CentroidY(is)/Area(is);
  Perim(is)=Perim(is)-2*pi*r0;
  % compute the moment M
  M(is)=0;
  for i=2:nx+1
      for j=2:ny+1
          M(is)=M(is)+chi(i,j)*((x(i)-x0)^2+(y(j)-y0)^2)*dx*dy;
      end
  end    

%------------------ Plot the results ---------------------------
  time=time+dt                   % plot the results
  uu(1:nx+1,1:ny+1)=0.5*(u(1:nx+1,2:ny+2)+u(1:nx+1,1:ny+1));
  vv(1:nx+1,1:ny+1)=0.5*(v(2:nx+2,1:ny+1)+v(1:nx+1,1:ny+1));
  for i=1:nx+1,xh(i)=dx*(i-1);end;     for j=1:ny+1,yh(j)=dy*(j-1);end
  hold off,contour(x,y,r'),axis equal,axis([0 Lx 0 Ly]);
  hold on;quiver(xh,yh,uu',vv','r');
  plot(xf(1:Nf),yf(1:Nf),'k','linewidth',2);pause(0.01)
end                  % End of time step

%------ Extra commands for interactive processing --------------
% plot(Time,Area,'r','linewidth',2); axis([0 dt*nstep 0 0.1]);
% set(gca,'Fontsize',18, 'LineWidth',2)
% T1=Time;A1=Area;CX1=CentroidX;CY1=CentroidY;
% T2=Time;A2=Area;CX2=CentroidX;CY2=CentroidY;
% figure, mesh(x,y,chi');

% quiver(nnx(2:nx,2:ny)',nny(2:nx,2:ny)');axis 'square'