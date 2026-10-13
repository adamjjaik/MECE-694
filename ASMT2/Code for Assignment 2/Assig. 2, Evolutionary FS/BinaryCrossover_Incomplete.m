function [Offspring1, Offspring2] = BinaryCrossover_Incomplete(Parent1, Parent2)
% This Function Performs Binary Crossover Operation Including
% Single Point Crossover, Double Point Crossover, Uniform Crossover

% One Crossover Operator Will be Selected In Each Iteration Using
% Roulette Wheel Selection

Offspring1 = [];
Offspring2 = [];

Prob_SPX = 0.15;
Prob_DPX = 0.25;
Prob_UX = 1 - Prob_SPX - Prob_DPX; % Less Probability for SPX and DPX
XMethod = RouletteWheelSelection([Prob_SPX Prob_DPX Prob_UX]);
switch XMethod
    case 1
        [Offspring1, Offspring2] = SinglePointCrossover(Parent1, Parent2);
    case 2
        % (!!!Complete This Function!!!)
        [Offspring1, Offspring2] = DoublePointCrossover(Parent1, Parent2);
    case 3
        % (!!!Complete This Function!!!)
        [Offspring1, Offspring2] = UniformCrossover(Parent1, Parent2);
end

% To Make Sure That At Least One Feature Is Selected
if sum(Offspring1) == 0 || sum(Offspring2) == 0
    if sum(Offspring1) == 0 % Fix Offspring 1
        Offspring1 = Parent1;
    else % Fix Offspring 2
        Offspring2 = Parent2;
    end
end
end

function [Offspring1, offspring2] = SinglePointCrossover(Parent1, Parent2)
% This Function Perform Single Point Crossover

% Select A Random Number
i1 = randi([1 length(Parent1)-1]);
% Perform Crossover
Offspring1 = [Parent1(1:i1) Parent2(i1+1:end)];
offspring2 = [Parent2(1:i1) Parent1(i1+1:end)];
end

% *********** WRITE YOUR CODE HERE *************
function [Offspring1, offspring2] = DoublePointCrossover(Parent1, Parent2)
% This Function Perform Double Point Crossover

% Select Two Random Numbers

% Perform Crossover

end

function [Offspring1, offspring2] = UniformCrossover(Parent1, Parent2)
% This Function Perform Uniform Crossover

% Create A Random Mask of 0s and 1s with Similar Length to Parents

% Perform Crossover

end
% **********************************************

function [Index] = RouletteWheelSelection(P)
% This function performs Roulette Wheel Selection
% based on given probabilities P

i = rand; % Select a random number
C = cumsum(P); % Cumulative sum of P
Index = find(i <= C, 1, 'first');
end









