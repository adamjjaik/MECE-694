%% Reference
% This Code was adopted from: 
% Mostapha Kalami Heris, NSGA-III: Non-dominated Sorting Genetic Algorithm, 
% the Third Version — MATLAB Implementation (URL: https://yarpiz.com/456/ypea126-nsga3), Yarpiz, 2016.
% Link: https://www.mathworks.com/matlabcentral/fileexchange/60678-nsga-iii-in-matlab?s_tid=prof_contriblnk

% This code was modifies for "MECE 788: Applied Computational Intelligence
% for Engineering" at the Univerisyt of Alberta, Winter 2021.

function UniqueSolutions = FindUniqueSolutions(F1)
% This Function Removes Members of Pareto Fron witi Similar Exactly Solution

% A Logical Vector to Point to Features that are Unique
UniqSol_Index = true(size(F1));

% Compare Each Two Memebrs of Pareto Front
for i = 2:size(F1,1)
    for j = 1:i-1
        % Are Solutions i and j Similar
        if all(F1(i).Solution == F1(j).Solution)
            UniqSol_Index(i) = false;
            break;
        end
    end
end

% Only Keep Unique Solutions
UniqueSolutions = F1(UniqSol_Index);
end
