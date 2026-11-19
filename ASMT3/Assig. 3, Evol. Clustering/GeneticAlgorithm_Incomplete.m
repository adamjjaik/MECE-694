%% Reference
% This Code was adopted from: 
% Mostapha Kalami Heris, Yarpiz Evolutionary Algorithms Toolbox for MATLAB (YPEA), Yarpiz, 2020.
% Link: https://www.mathworks.com/matlabcentral/fileexchange/71373-ypea?s_tid=prof_contriblnk

% This code was modifies for "MECE 788: Applied Computational Intelligence
% for Engineering" at the Univerisyt of Alberta, Winter 2021.

%% (1) Create Data for Clustering
clear
clc

% x: input data --> M-by-p (M: # of observations, p: # of features)
Model = 'Model_4';
load(Model)
PlotTrueClusters(Model);

% Set the number of clusters to a large number initially
NumCluster = 10;

clearvars -except x NumCluster t

%% (2) Define Clustering Objective Function and Decision Variables Info
%%%% Two Objectives were defined for the clustering problem
% WCD: minimizng the within class distance (classic objective function)
% DB_Index: ratio of "withing class distance" over "between class distance"
% Use Parameter "Method" to identify the objective you want to use
Method = 'DB_Index';

% Clustering Objective Function
Objective = @(C) ClusteringObjValue(C, x, Method);    

% Size of Decision Variable Matrix (# of Clusters, # of Dec Variables)
% VarSize = ;  % !!! Complete This Section !!!

% Number of Decision Variables
NumVar = prod(VarSize);     

% Lower and Upper Bounds of Decision Variables
% MinimumVar = ; % !!! Complete This Section !!!
% MaximumVar = ; % !!! Complete This Section !!!     

%% (3) Genetic Algorithm Parameters % !!! Complete This Section !!!


%% (4) Initialize Genetic Algorithm % !!! Complete This Section !!!


%% (5) Genetic Algorithm Main Loop % !!! Complete This Section !!!


%% Plot Objective Value versu Iteration Number
figure(3)
plot(1:MaxIter, BestObjValues, 'LineWidth', 2);
xlabel('Iteration Number'), ylabel('Objective Value')
title('Evolutionary Clustering')
