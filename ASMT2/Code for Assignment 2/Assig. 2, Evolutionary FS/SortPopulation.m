%% Reference
% This Code was adopted from: 
% Mostapha Kalami Heris, NSGA-III: Non-dominated Sorting Genetic Algorithm, 
% the Third Version — MATLAB Implementation (URL: https://yarpiz.com/456/ypea126-nsga3), Yarpiz, 2016.
% Link: https://www.mathworks.com/matlabcentral/fileexchange/60678-nsga-iii-in-matlab?s_tid=prof_contriblnk

% This code was modifies for "MECE 788: Applied Computational Intelligence
% for Engineering" at the Univerisyt of Alberta, Winter 2021.

function [Pop, F] = SortPopulation(Pop)
% Function Sorts Pop Members Based on Crowding Distance and Rank 


% Sorting Based on Crowding Distance
[~, CDSortOrder] = sort([Pop.CrowdingDistance], 'descend');
Pop = Pop(CDSortOrder);

% Sorting Based on Rank
[~, RSortOrder] = sort([Pop.Rank]);
Pop = Pop(RSortOrder);

% Updating All Fronts
Ranks = [Pop.Rank];
MaximumRank = max(Ranks);

% Maximum Rank Determines Number of Fronts
F = cell(MaximumRank,1);
% Assign Solutions with Rank r to Front r
for r = 1:MaximumRank
    F{r} = find(Ranks == r);
end
end