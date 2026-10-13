%% Reference
% This Code was adopted from: 
% Mostapha Kalami Heris, Yarpiz Evolutionary Algorithms Toolbox for MATLAB (YPEA), Yarpiz, 2020.
% Link: https://www.mathworks.com/matlabcentral/fileexchange/71373-ypea?s_tid=prof_contriblnk

% This code was modifies for "MECE 788: Applied Computational Intelligence
% for Engineering" at the Univerisyt of Alberta, Winter 2021.


function PlotSolution(Solution, Model)
% This Function Plots the Solution of TSP 

Tour = Solution.Tour;
Tour = [Tour Tour(1)];
plot(Model.x(Tour), Model.y(Tour),'b-p',...
    'LineWidth',2,...
    'MarkerSize',12,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r');

xlim([Model.xmin Model.xmax]);
ylim([Model.ymin Model.ymax]);
end