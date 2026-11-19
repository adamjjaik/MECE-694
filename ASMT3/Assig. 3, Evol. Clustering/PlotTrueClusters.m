%% Reference
% This Code was adopted from: 
% Mostapha Kalami Heris, Yarpiz Evolutionary Algorithms Toolbox for MATLAB (YPEA), Yarpiz, 2020.
% Link: https://www.mathworks.com/matlabcentral/fileexchange/71373-ypea?s_tid=prof_contriblnk

% This code was modifies for "MECE 788: Applied Computational Intelligence
% for Engineering" at the Univerisyt of Alberta, Winter 2021.

function PlotTrueClusters(Model)
% This function plots the data points and their true clusters

load(Model) 
% Plot Data
figure(1)
Colors = hsv(NumCluster);
% Plot data points associated with each cluster
for i = 1:NumCluster
    plot(x(t==i,1), x(t==i,2), 'p', 'LineWidth', 2, 'Color', Colors(i,:));
    hold on;
    % Plot Cluster Centers
    plot(mean(x(t==i,1)), mean(x(t==i,2)), '*k','LineWidth',2, 'MarkerSize', 12);
end
hold off;
ylabel('Feature: x_2'), xlabel('Feature: x_1')
title(['True Clusters  ,  ' Model]), 
axis([min(x(:,1))-1 max(x(:,1))+1 min(x(:,2))-1 max(x(:,2))+1])
end

