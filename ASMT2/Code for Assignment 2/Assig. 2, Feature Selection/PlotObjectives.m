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