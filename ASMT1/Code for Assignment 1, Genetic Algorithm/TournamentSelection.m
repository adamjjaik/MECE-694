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