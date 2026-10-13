%% Reference
% This Code was adopted from: 
% Mostapha Kalami Heris, Yarpiz Evolutionary Algorithms Toolbox for MATLAB (YPEA), Yarpiz, 2020.
% Link: https://www.mathworks.com/matlabcentral/fileexchange/71373-ypea?s_tid=prof_contriblnk

% This code was modifies for "MECE 788: Applied Computational Intelligence
% for Engineering" at the Univerisyt of Alberta, Winter 2021.

function [Offspring1, Offspring2] = ArithmeticCrossover(...
    parent1, parent2, gamma, VarMin, VarMax)

% This function perform arithmetic crossover

    alpha = unifrnd(-gamma, 1+gamma, size(parent1));
    
    % Calculating Offspring
    Offspring1 = alpha .* parent1 + (1-alpha) .* parent2;
    Offspring2 = alpha .* parent2 + (1-alpha) .* parent1;
    
    % Ensuring Offspring are in Acceptable Range
    Offspring1 = max(Offspring1, VarMin);
    Offspring1 = min(Offspring1, VarMax);
    
    Offspring2 = max(Offspring2, VarMin);
    Offspring2 = min(Offspring2, VarMax);
end