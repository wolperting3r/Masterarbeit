%% Auswertung y-Position der Oszillation nach Strubelj temp
clear all
clc

%% Read reference data
% 128 cvofls
for sims=1:1
pfad='/work/local/friedrich/06_FASTEST/04_Codes/01_Fastest/fastest/projects/02_Oscillation/res/single/ori_cvofls_128_paper/';
datei='y_pos.txt';
filename=[pfad datei];
headerlinesIn=1;
delimiterIn=' ';
data=importdata(filename,delimiterIn,headerlinesIn);
[laenge reihen]=size(data.data);
it=data.data(:,1);
y=data.data(:,3);
x=data.data(:,2);
conc=data.data(:,4);
zeitschrittref=2.5d-3;

ystart=data.data(1,3);
diffpos=0;
for n=1:laenge
    if data.data(1,1)==data.data(n,1)
        diffpos=diffpos+1;  
    else break        
    end 
end

ycells=diffpos;
count=1;
count2=1;
counter=laenge/ycells;

for i=1:counter
    for j=1:ycells
        concs(j,i)=conc(j+ycells*(i-1));
        yposes(j,i)=y(j+ycells*(i-1));
    end
end
for i=1:counter
    concmin(count)=0.99;
    for j=1:ycells
        if concs(j,i)<1
            if concs(j,i)>0
                if concmin(count) > concs(j,i)
                    concmin(count)=concs(j,i);
                    ymin(count)=yposes(j,i);
                end
            end
        end
    end
    count=count+1;
end

for i=1:counter
    concmax(count2)=0.01;
    for j=1:ycells
        if concs(j,i)<1
            if concs(j,i)>0
                if concmax(count2) < concs(j,i)
                    concmax(count2)=concs(j,i);
                    ymax(count2)=yposes(j,i);
                end
            end
        end
    end
    count2=count2+1;
end

%% Interpolation der Verläufe
% Werte dürfen nur einmal vorkommen im Array!
tol=1e-3;
array=1;
for tref=1:counter
    conc_reduced(1)=1;
    y_reduced(1)=0.0519;
    counter_reduced=1;
    for tt=1:ycells        
        if conc_reduced(counter_reduced)-concs(tt,tref)>tol
            counter_reduced=counter_reduced+1;
            conc_reduced(counter_reduced)=concs(tt,tref);
            y_reduced(counter_reduced)=yposes(tt,tref);
        end
    end
    concarray(tref).concs=[conc_reduced(:)];
    concarray(tref).y=[y_reduced(:)];
    concarray(tref).interpol=interp1(concarray(tref).concs, concarray(tref).y, 0.5);
    interpolands_128_paper(tref)=concarray(tref).interpol;
    clear conc_reduced
    clear y_reduced
end

end

%strubelj 128
pfad='/work/local/friedrich/06_FASTEST/04_Codes/01_Fastest/fastest/projects/02_Oscillation/res/ref_strubelj/';
datei='Strubelj_128.txt';
filename=[pfad datei];
headerlinesIn=1;
delimiterIn=' ';
data=importdata(filename,delimiterIn,headerlinesIn);
[laenge reihen]=size(data.data);
x_ref=data.data(:,1);
y_ref=data.data(:,2);

%128 51 without exponent paper
for sims=1:1
pfad='/work/local/friedrich/06_FASTEST/04_Codes/01_Fastest/fastest/projects/02_Oscillation/res/single/51_withoutexponent_128_paper/';
%pfad='/work/local/friedrich/06_FASTEST/04_Codes/01_Fastest/fastest/projects/02_Oscillation/res/128_celeste/';
datei='y_pos.txt';
filename=[pfad datei];
headerlinesIn=1;
delimiterIn=' ';
data=importdata(filename,delimiterIn,headerlinesIn);
[laenge reihen]=size(data.data);
it=data.data(:,1);
y=data.data(:,3);
x=data.data(:,2);
conc=data.data(:,4);
zeitschritt=2.5d-3;

ystart=data.data(1,3);
diffpos=0;
for n=1:laenge
    if data.data(1,1)==data.data(n,1)
        diffpos=diffpos+1;  
    else break        
    end 
end

ycells=diffpos;
count=1;
count2=1;
counter=laenge/ycells;

for i=1:counter
    for j=1:ycells
        concs(j,i)=conc(j+ycells*(i-1));
        yposes(j,i)=y(j+ycells*(i-1));
    end
end
for i=1:counter
    concmin(count)=0.99;
    for j=1:ycells
        if concs(j,i)<1
            if concs(j,i)>0
                if concmin(count) > concs(j,i)
                    concmin(count)=concs(j,i);
                    ymin(count)=yposes(j,i);
                end
            end
        end
    end
    count=count+1;
end

for i=1:counter
    concmax(count2)=0.01;
    for j=1:ycells
        if concs(j,i)<1
            if concs(j,i)>0
                if concmax(count2) < concs(j,i)
                    concmax(count2)=concs(j,i);
                    ymax(count2)=yposes(j,i);
                end
            end
        end
    end
    count2=count2+1;
end

%% Interpolation der Verläufe
% Werte dürfen nur einmal vorkommen im Array!
tol=1e-3;
array=1;
for t=1:counter
    conc_reduced(1)=1;
    y_reduced(1)=0.0519;
    counter_reduced=1;
    for tt=1:ycells        
        if conc_reduced(counter_reduced)-concs(tt,t)>tol
            counter_reduced=counter_reduced+1;
            conc_reduced(counter_reduced)=concs(tt,t);
            y_reduced(counter_reduced)=yposes(tt,t);
        end
    end
    concarray(t).concs=[conc_reduced(:)];
    concarray(t).y=[y_reduced(:)];
    concarray(t).interpol=interp1(concarray(t).concs, concarray(t).y, 0.5);
    interpolands_128_51without(t)=concarray(t).interpol;
    tref2=t;
    clear t
    clear conc_reduced
    clear y_reduced
end

end
clear concarray
clear conc
clear concs
clear concmin
clear concmax
%% Read sim data
% sim
for sims=1:1
pfad='/work/local/friedrich/06_FASTEST/04_Codes/01_Fastest/fastest/projects/02_Oscillation/';%res/single/51_withoutexponent_256_paper_secondweighting/';
datei='y_pos.txt';
filename=[pfad datei];
headerlinesIn=1;
delimiterIn=' ';
data=importdata(filename,delimiterIn,headerlinesIn);
[laenge reihen]=size(data.data);
it=data.data(:,1);
y=data.data(:,3);
x=data.data(:,2);
conc=data.data(:,4);
zeitschrittsim=1d-2;

ystart=data.data(1,3);
diffpos=0;
for n=1:laenge
    if data.data(1,1)==data.data(n,1)
        diffpos=diffpos+1;  
    else break        
    end 
end

ycells=diffpos;
count=1;
count2=1;
counter=laenge/ycells;
tol=1e-3;
% iterate over all time steps
for i=1:counter
    % iterate over all y-positions in time step
    for j=1:ycells
        concs(j,i)=conc(j+ycells*(i-1));
        yposes(j,i)=y(j+ycells*(i-1));
        % correct small numbers
        if ( concs(j,i) > (1-tol))
          concs(j,i)=1;
        elseif ( concs(j,i) < tol )
          concs(j,i)=0;
        end
    end
end
 
% iterate over all time steps
for i=1:counter
    concmin(count)=0.99;
    % iterate over all y-positions in time step
    for j=1:ycells
	% find minimal c and corresponding y-pos for c>0, c<1
        if concs(j,i)<1
            if concs(j,i)>0
                if concmin(count) > concs(j,i)
                    concmin(count)=concs(j,i);
                    ymin(count)=yposes(j,i);
                end
            end
        end
    end
    count=count+1;
end

% iterate over all time steps
for i=1:counter
    concmax(count2)=0.01;
    % iterate over all y-positions in time step
    for j=1:ycells
	% find maximum c and corresponding y-pos for c<1
        if concs(j,i)<1
            if concs(j,i)>0
                if concmax(count2) < concs(j,i)
                    concmax(count2)=concs(j,i);
                    ymax(count2)=yposes(j,i);
                end
            end
        end
    end
    count2=count2+1;
end

%% Interpolation der Verläufe
% Werte dürfen nur einmal vorkommen im Array!
tol=1e-3;
array=1;
% Iterate over all time steps
for t=1:counter
    conc_reduced(1)=1;
    y_reduced(1)=0.0519;
    counter_reduced=1;
    % Iterate over all y-positions in time step
    for tt=1:ycells        
	% Wenn die Differenz zum letzten Wert > tol ist, dann hänge den Wert an. Entfernt doppelte (innerhalb der Toleranz) Werte.
        if conc_reduced(counter_reduced)-concs(tt,t)>tol
            counter_reduced=counter_reduced+1;
            conc_reduced(counter_reduced)=concs(tt,t);
            y_reduced(counter_reduced)=yposes(tt,t);
        end
    end
    concarray(t).concs=[conc_reduced(:)];
    concarray(t).y=[y_reduced(:)];
    % lineare Interpolation des y-Wertes wo c = 0.5 liegt
    concarray(t).interpol=interp1(concarray(t).concs, concarray(t).y, 0.5);
    interpolands_128_sim(t)=concarray(t).interpol;
    tsim=t;
    clear conc_reduced
    clear y_reduced
    clear t
end

end



%% Figures
close all

figure(1)
hold on
plot((1:(tref))*zeitschrittref,interpolands_128_paper,'b')
plot((1:(tref2))*zeitschrittref,interpolands_128_51without,'k')
plot((1:(tsim))*zeitschrittsim,interpolands_128_sim,'g')
legend('cvofls','51without','sim')

%plot(x_ref,y_ref,'--b')
%plot((1:(tref))*zeitschrittref,interpolands_128_paper,'b')
%plot((1:(tref2))*zeitschritt,interpolands_128_51without,'k')
%plot((1:(tsim))*zeitschrittsim,interpolands_128_sim,'r')
%legend('strubelj','cvofls','51without','sim')
