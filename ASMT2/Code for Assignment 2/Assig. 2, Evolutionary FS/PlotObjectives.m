%% Reference
% This Code was adopted from: 
% Mostapha Kalami Heris, NSGA-III: Non-dominated Sorting Genetic Algorithm, 
% the Third Version — MATLAB Implementation (URL: https://yarpiz.com/456/ypea126-nsga3), Yarpiz, 2016.
% Link: https://www.mathworks.com/matlabcentral/fileexchange/60678-nsga-iii-in-matlab?s_tid=prof_contriblnk

% This code was modifies for "MECE 788: Applied Computational Intelligence
% for Engineering" at the Univerisyt of Alberta, Winter 2021.

function PlotObjectives(Pop)
% This Function Plots Objectives at Each Iteration

ObjectiveValue = [Pop.ObjValue];
plot(ObjectiveValue(1,:), ObjectiveValue(2,:), ...
    'r*','MarkerSize',8);
xticks(min(ObjectiveValue(1,:))-1:1:max(ObjectiveValue(1,:))+1);
xlabel('Number of Selected Features'); ylabel('Train + Test  Errors');
title('Pareto Front: Evolutionary Feature Selection')
grid on
end