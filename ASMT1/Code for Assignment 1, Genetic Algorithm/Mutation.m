%% Reference
% This Code was adopted from: 
% Mostapha Kalami Heris, Yarpiz Evolutionary Algorithms Toolbox for MATLAB (YPEA), Yarpiz, 2020.
% Link: https://www.mathworks.com/matlabcentral/fileexchange/71373-ypea?s_tid=prof_contriblnk

% This code was modifies for "MECE 788: Applied Computational Intelligence
% for Engineering" at the Univerisyt of Alberta, Winter 2021.

function [offspring] = Mutation(parent, MutationRate, VarMin, VarMax)
% This function perform Mutation

% Determine # of elements in parent
nVar = numel(parent);

% # of elements under mutation
nmutation = ceil(MutationRate * nVar);

% Randomly select nmutation elements in parent
j = randsample(nVar, nmutation);

% Perform Mutation
sigma = 0.1*(VarMax-VarMin);
offspring = parent;
offspring(j) = parent(j) + sigma * randn(size(j));

% Ensuring Offspring are in Acceptable Range
offspring = max(offspring, VarMin);
offspring = min(offspring, VarMax);
end