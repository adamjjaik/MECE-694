function [Offspring1, Offspring2] = NewCrossover(c1, c2, alpha, VarMin, VarMax)
% BLX-alpha (Blended Crossover), Eshelman & Schaffer (1993)
% Reference: Herrera et al., "Multiple Crossover per Couple with Selection
% of the Two Best Offspring: An Experimental Study with the BLX-alpha
% Crossover Operator for Real-Coded Genetic Algorithms," IBERAMIA 2002.

% Determine the min, max, and range between the two parents
c_min = min(c1, c2);
c_max = max(c1, c2);
I = c_max - c_min;

% Generate each offspring by drawing uniformly from the extended interval
% [c_min - I*alpha, c_max + I*alpha], independently for each gene
lower_bound = c_min - I*alpha;
upper_bound = c_max + I*alpha;

Offspring1 = lower_bound + rand(size(c1)) .* (upper_bound - lower_bound);
Offspring2 = lower_bound + rand(size(c1)) .* (upper_bound - lower_bound);

% Ensuring Offspring are in Acceptable Range
Offspring1 = max(Offspring1, VarMin);
Offspring1 = min(Offspring1, VarMax);
Offspring2 = max(Offspring2, VarMin);
Offspring2 = min(Offspring2, VarMax);

end