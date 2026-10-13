%% Reference
% This Code was adopted from: 
% Mostapha Kalami Heris, NSGA-III: Non-dominated Sorting Genetic Algorithm, 
% the Third Version — MATLAB Implementation (URL: https://yarpiz.com/456/ypea126-nsga3), Yarpiz, 2016.
% Link: https://www.mathworks.com/matlabcentral/fileexchange/60678-nsga-iii-in-matlab?s_tid=prof_contriblnk

% This code was modifies for "MECE 788: Applied Computational Intelligence
% for Engineering" at the Univerisyt of Alberta, Winter 2021.

function [Pop] = CalculateCrowdingDistance(Pop, F)
% This Function Calculates Crowding Distance For 
% Pop Memebers in Each Front k

for k = 1:size(F,2)
    % Extract the Objectve Values for Memebrs in Front k
    ObjValues = [Pop(F{k}).ObjValue];
    % Determine the Number of Objective Functions
    NumObjective = size(ObjValues,1);
    % Determine the Number of Elements in Front k
    NumElements_k = numel(F{k});
    
    % Create an Array to Hold Crowding Distance
    d = zeros(NumElements_k, NumObjective);
    % For Each Objective Function j
    for j = 1:NumObjective
        % Sort Objectve Values
        [ObjSorted, SortOrder] = sort(ObjValues(j,:));
        
        % Crowding Distance for Member with Lowest Objective j is Infinity
        d(SortOrder(1),j) = inf;
        
        % For Other Members, Calculate Crowding Distance
        for i = 2:NumElements_k - 1
            d(SortOrder(i),j) = ...
                abs(ObjSorted(i+1) - ObjSorted(i-1))/ ...
                abs(ObjSorted(1) - ObjSorted(end));
        end
        
        % Crowding Distance for Member with Highest Objective j is Infinity
        d(SortOrder(end),j) = inf;
    end
    
    % Assign the Calculated Crowding Distance to Pop Members in Front k
    for i = 1:NumElements_k
        Pop(F{k}(i)).CrowdingDistance = sum(d(i,:));
    end
end
end

