% =============================================================
% ML-Train.m
% A code to solve an input-output problem with the Neural Network
% tool box in MATLAB. One may also do the same thing using GUI by 
% type "nftool" in the command window. The code assumes two 
% variables are defined: input - input data and output - output 
% data.
% Author: Yinghe Qi and Gretar Tryggvason (5/25/2018)
% =============================================================

clear
close all

load database.mat
x = datafra';
t = datacur';

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

% Create a Fitting Network
hiddenLayerSize = 100;
net = fitnet(hiddenLayerSize,trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y)

% % View the Network
% view(net)

% Generate function
genFunction(net,'NNCircle2');

% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, ploterrhist(e)
figure, plotregression(t,y)
% figure, plotfit(net,x,t)
