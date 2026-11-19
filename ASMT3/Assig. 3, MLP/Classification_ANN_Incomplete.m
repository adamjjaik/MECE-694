%% (1) Create Data
clear
clc
close all

% x: input data --> p by M (M: # of observations, p: # of features)
% Class 1
Mu1 = [5 5]; Sigma1 = [3  0.1; 0.1  3];
X1 = mvnrnd(Mu1, Sigma1, 50)';

Mu1 = [6 -2]; Sigma1 = [2  0; 0  2];
X1 = [X1 mvnrnd(Mu1, Sigma1, 50)'];

% Class 2
Mu2 = [1 1]; Sigma2 = [5  0.5; 0.5  5];
X2 = mvnrnd(Mu2, Sigma2, 100)';

x = [X1'; X2']';
t_org = [ones(size(X1,2),1); -1*ones(size(X2,2),1)]';

figure(1)
plot(X1(1,:), X1(2,:), 'b*', 'DisplayName','Class 1');
hold on
plot(X2(1,:), X2(2,:), 'r+', 'DisplayName','Class 2');
hold off
ylabel('x_2'), xlabel('x_1')
axis([-4 12 -4 11])
legend

%% (2) Feedforward Nerual Network (MLP)
% Creating proper output format for MLP
t = [t_org; t_org];
t(1,t(1,:) == -1) = 0;
t(2,t(2,:) == 1) = 0; t(2,t(2,:) == -1) = 1;

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
% 'traingdm'
% 'traingda'
% 'traingdx'
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

%%%% Create a Network
hiddenLayerSize = 10;
% Use 'fitnet' Function
net = fitnet(hiddenLayerSize,trainFcn);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 100/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 0/100;

% Choose a Performance Function

net.performFcn = 'mse';  % Mean Squared Error

% Choose Plot Functions
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};

% Set Advanced Training Parameters
net.trainParam.showWindow = true;
net.trainParam.showCommandLine = false;
net.trainParam.epochs = 1000;
net.trainParam.goal = 0;

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);

% Plot
figure, plotconfusion(t, y, 'All Data');
title('Performance of MATLAB MLP Implementation')

clearvars -except x t_org

%% (3) MLP Training with Backpropagation, Your Code
% Creating proper output format for MLP
t = [t_org; t_org];
t(1,t(1,:) == -1) = 0;
t(2,t(2,:) == 1) = 0; t(2,t(2,:) == -1) = 1;

%%%% Initialize Parameters
% Number of Features
NumFeatures  = size(x,1);
% Number of artificial neurons in the hidden layer
hiddenLayerSize = 10;
% Number of Classes
NumClasses = 2;


%%% In this section you will implement a two layer MLP for classification
% STEP 1: Complete function named " RandomInitialWeights_Incomplete" to initialize MLP weights randomly
Initial_theta1 = RandomInitialWeights_Incomplete(NumFeatures, hiddenLayerSize);
Initial_theta2 = RandomInitialWeights_Incomplete(hiddenLayerSize, NumClasses);

% STEP 2:Unroll Initial Parameters
ANN_Initial_Weights = [Initial_theta1(:) ; Initial_theta2(:)];

% STEP 3: Implement a function to perfome "feedforward" and "back" propagation
% using MLP wieghts, "ANN_Weights", and return objective value and gradients
lambda = 0; % Regularization Weight

[J, GradJ] = MyANNObjecFunction_Incomplete(ANN_Initial_Weights, NumFeatures, hiddenLayerSize, ...
    NumClasses, x, t, lambda);

% At this point, we have all the functions required for training MLP with
% Back-Propagation Algrithm.

% STEP 4: Use "fminunc" to find MLP wieghts which minimizes the objectve
% function value
options = optimoptions('fminunc', ...
    'Algorithm','trust-region', ... % choices are: quasi-newton OR trust-region
    'SpecifyObjectiveGradient', true, ...
    'Display','iter', ...
    'MaxIterations', 150);
% Objective Function for "fminunc"
Objective = @(ANN_Weights) MyANNObjecFunction_Incomplete(ANN_Weights, ...
    NumFeatures, hiddenLayerSize, NumClasses, x, t, lambda);

% Apply "fminunc" to find optimal MLP parameters
ANN_Weights = fminunc(Objective, ANN_Initial_Weights, options);

% STEP 5: Obtain input-to-hidden layer weight (theta1) hidden-to-output
% layer weight (theta2) matrix fron unrolled ANN parameters
theta1 = reshape(ANN_Weights(1:hiddenLayerSize * (NumFeatures + 1)), ...
    hiddenLayerSize, (NumFeatures + 1));
theta2 = reshape(ANN_Weights((1 + (hiddenLayerSize * (NumFeatures + 1))):end), ...
    NumClasses, (hiddenLayerSize + 1));

% Step 6: Estimate Outputs Using the Obtained Optimized MLP Weights
y = PredictMLPOutputs(theta1, theta2, x);

% Plot
figure, plotconfusion(t, y, 'All Data')
title('Performance of Your MLP Implementation')

clearvars -except x t_org

