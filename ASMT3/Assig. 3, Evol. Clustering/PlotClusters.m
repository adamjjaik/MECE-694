%% Reference
% This Code was adopted from: 
% Mostapha Kalami Heris, Yarpiz Evolutionary Algorithms Toolbox for MATLAB (YPEA), Yarpiz, 2020.
% Link: https://www.mathworks.com/matlabcentral/fileexchange/71373-ypea?s_tid=prof_contriblnk

% This code was modifies for "MECE 788: Applied Computational Intelligence
% for Engineering" at the Univerisyt of Alberta, Winter 2021.

function PlotClusters(x, BestSolution)
% This function plots the clustering results for data in "x" and cluster
% centers in "BestSolution.Solution"


% Extract cluster centers from "BestSolution"
C = BestSolution.AddInfo.C;

% Extract Number of Clusters
NumCluster = size(C,1);

% Extract Cluster Indices: which data point belongs to which cluster
Ind = BestSolution.AddInfo.Ind_d;

% Plot data points associated with each cluster
Colors = hsv(NumCluster);
for i = 1:NumCluster
    Xj = x(Ind == i, :);
    plot(Xj(:,1), Xj(:,2), 'p', 'LineWidth', 2, 'Color', Colors(i,:));
    hold on;
end
% Plot Cluster Centers
plot(C(:,1), C(:,2), '*k','LineWidth',2, 'MarkerSize', 12);
hold off
ylabel('Feature: x_2'), xlabel('Feature: x_1')
title('Evolutionary Clustering'), 
axis([min(x(:,1))-1 max(x(:,1))+1 min(x(:,2))-1 max(x(:,2))+1])
end