function [Offspring1, Offspring2] = HeuristicCrossover(...
    parent1, parent2, parent3, parent4, VarMin, VarMax)

% This function perform a new crossover implemented by you

    % Find a Random Number (0 <= alpha <= 1)
    %%% Write your code here
    alpha = rand;
    % Find Best and Worst Parents (x_star and x_plus)
    %%% Write your code here
    % Combine the four parents into an array so they can
    % be indexed together by position (1 through 4)
    parents = [parent1, parent2, parent3, parent4];

    % Collect each parent's objective value into one array, in the
    % same order as 'parents', so positions line up for indexing
    objvals = [parent1.ObjValue, parent2.ObjValue, parent3.ObjValue, parent4.ObjValue];

    % Find the position (not just the value) of the smallest and
    % largest objective value among the four parents
    [~, idx_best]  = min(objvals);
    [~, idx_worst] = max(objvals);

    % Use those positions to extract the actual best (x_star) and
    % worst (x_plus) parent solutions for the offspring equations
    x_star = parents(idx_best).Solution;
    x_plus = parents(idx_worst).Solution;

    % Calculating Offspring
    %%% Write your code here
    Offspring1 = x_star + alpha*(x_star - x_plus);
    Offspring2 = x_plus + alpha*(x_star - x_plus);
    % Ensuring Offspring are in Acceptable Range
    %%% Write your code here
    Offspring1 = max(Offspring1, VarMin);
    Offspring1 = min(Offspring1, VarMax);
    Offspring2 = max(Offspring2, VarMin);
    Offspring2 = min(Offspring2, VarMax);

end