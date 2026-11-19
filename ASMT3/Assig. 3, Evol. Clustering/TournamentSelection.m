%% Reference
% This Code was adopted from: 
% Mostapha Kalami Heris, Yarpiz Evolutionary Algorithms Toolbox for MATLAB (YPEA), Yarpiz, 2020.
% Link: https://www.mathworks.com/matlabcentral/fileexchange/71373-ypea?s_tid=prof_contriblnk

% This code was modifies for "MECE 788: Applied Computational Intelligence
% for Engineering" at the Univerisyt of Alberta, Winter 2021.

function [Index] = TournamentSelection(Pop, TournamentSize)
% This function performs Tournament Selection


% Select TournamentSize samples from polulation randomly
SIndex = randsample(numel(Pop),TournamentSize);
SelectedPop = Pop(SIndex);

% Sort the Select Parents
SelectedObjValue = [SelectedPop.ObjValue];

% Return Index of Parent with Minimum ObjValue
[~, j] = min(SelectedObjValue);
Index = SIndex(j);
end