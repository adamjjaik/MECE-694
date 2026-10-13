%% Reference
% This Code was adopted from: 
% Mostapha Kalami Heris, NSGA-III: Non-dominated Sorting Genetic Algorithm, 
% the Third Version — MATLAB Implementation (URL: https://yarpiz.com/456/ypea126-nsga3), Yarpiz, 2016.
% Link: https://www.mathworks.com/matlabcentral/fileexchange/60678-nsga-iii-in-matlab?s_tid=prof_contriblnk

% This code was modifies for "MECE 788: Applied Computational Intelligence
% for Engineering" at the Univerisyt of Alberta, Winter 2021.

function [Pop, F] = NonDominatedSorting(Pop)
% This Function Performs Non-Dominated Sorting for NSGA-II
%

% STEP 1: Clear the Domination Parameters
% DominationSet: Memebr of Pop which are dominated by i-th Memebr
% DominatedCount: Number of Times other Pop Memebers Dominated i-th Memebr
for i = 1:size(Pop,1)
    Pop(i).DominationSet = [];
    Pop(i).DominatedCount = 0;
end

% Pareto Front (Empty at First)
F{1} = [];

% STEP 2: Compare All Pop Memeber two-by-two
for i = 1:size(Pop,1) % For Memeber i-th
    for j = i+1:size(Pop,1) % For Memeber j-th
        p = Pop(i);
        q = Pop(j);
        
        % STEP 2-1: If p Dominates q --> add q to p.DominationSet
        % and Add 1 to q.DominatedCount
        if Dominates(p.ObjValue, q.ObjValue)
            p.DominationSet = [p.DominationSet j];
            q.DominatedCount = q.DominatedCount + 1;
        end
        
        % If q Dominates p --> add p to q.DominationSet
        % and Add 1 to p.DominatedCount
        if Dominates(q.ObjValue, p.ObjValue)
            q.DominationSet = [q.DominationSet i];
            p.DominatedCount = p.DominatedCount+1;
        end
        
        % Update p and q In Pop
        Pop(i) = p;
        Pop(j) = q;
    end
    
    % STEP 3: If i-th Member Is NOT Dominated by Anyother Solution
    % (1) Add it for Pareto Front; (2) Give it Ranke = 1 
    if Pop(i).DominatedCount == 0
        F{1} = [F{1} i];
        Pop(i).Rank = 1;
    end
end

% STEP 4: Update Pareto Fron Number to 1
k = 1;

while true
    % STEP 5: Consider Q to be Empty
    Q = [];
    
    % STEP 6: For Every Member of Fk such as p
    for i = F{k}
        p = Pop(i);
         % STEP 6-1: For Every Member of p.DominationSet such as q
         % Decrease q.DominatedCount by 1
         % If q.DominatedCount == 0, Add q to Q
         % Increase q.Rank by 1
        for j = p.DominationSet
            q = Pop(j);
            q.DominatedCount = q.DominatedCount - 1;
            if q.DominatedCount == 0
                Q = [Q j];
                q.Rank = k+1;
            end
            % Update q
            Pop(j) = q;
        end
    end
    
    % STEP 7: If Q is Empty, Sorting is Finished
    if isempty(Q)
        break;
    end
    
    % STEP 8: If Q is NOT Empty, 
    % STEP 8-1: Fk+1 = Q (Front Number k+1)
    F{k+1} = Q;
    
    % STEP 8-2: Update Front Number
    k = k+1;
end
end

function flag = Dominates(Solution1, Solution2)
% Check If Solution1 Dominates Solution2
% flag: 0 --> NOT dominated  and 1 --> dominated

flag = all(Solution1 <= Solution2) && any(Solution1 < Solution2);
end

