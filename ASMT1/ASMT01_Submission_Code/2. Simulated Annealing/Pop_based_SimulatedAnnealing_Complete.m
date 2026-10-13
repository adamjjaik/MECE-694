%% Reference
% This Code was adopted from:
% Mostapha Kalami Heris, Yarpiz Evolutionary Algorithms Toolbox for MATLAB (YPEA), Yarpiz, 2020.
% Link: https://www.mathworks.com/matlabcentral/fileexchange/71373-ypea?s_tid=prof_contriblnk
% This code was modifies for "MECE 788: Applied Computational Intelligence
% for Engineering" at the Univerisyt of Alberta, Winter 2021.
%% Initialization: Problem Definition
clc
clear
% Create Problem Model
NumberCities = 30;
Model = CreateModel('Model 2', NumberCities);
% Cost Function
ObjectiveFunction = @(tour) TourLength(tour, Model);
%% Simulate Annealing Parameters
% Maximum Number of Iterations
MaxIter = 1000;
% Maximum Number of Sub-iterations
MaxSubIter = 15;
% Initial Temperature
T = 1;
% Temperature Reduction Rate
alpha = 0.90;
% Population Size
NumPop = 3;
% Number of Neighbors per Individual
NumNeigh = 5;
%% Simulate Annealing Initialization
% Create an Empty Structure for Solutions
empty_solutions.Tour = [];
empty_solutions.TourLength = [];
% Create Population Array
Pop = repmat(empty_solutions, NumPop, 1);
% Initialize Best Solution
BestSolution.TourLength = inf;
% Initialize Population
for i = 1:NumPop
% Create an Initial Solution
    Pop(i).Tour = CreateRandomTour(Model);
% Evaluate the Initial Solution
    Pop(i).TourLength = ObjectiveFunction(Pop(i).Tour);
% Update Best Solution
if Pop(i).TourLength <= BestSolution.TourLength
        BestSolution = Pop(i);
end
end
% Array to Hold Best TourLength Values
BestTourLength = zeros(MaxIter,1);
%% Simulate Annealing Main Loop
for iteration = 1:MaxIter
for subiteration = 1:MaxSubIter
% Create and Evaluate New Solutions
        NewPop = repmat(empty_solutions, NumPop, NumNeigh);
for i = 1:NumPop
for j = 1:NumNeigh
% Create a new Neighbor
% (COMPLETE THIS: Write your code here)
    % Generate a neighboring tour from population member i's
    % current solution using the given swap/reversion/insertion
    % operators (handled inside CreateNeighbour)
    NewPop(i,j).Tour = CreateNeighbour(Pop(i).Tour);
% (COMPLETE THIS!) Evaluation
% (COMPLETE THIS: Write your code here)
    % Evaluate the length of this newly generated neighbor tour
    NewPop(i,j).TourLength = ObjectiveFunction(NewPop(i,j).Tour);
end
end
        NewPop = NewPop(:);
% Sort Neighbors
        [~, SortOrder] = sort([NewPop.TourLength]);
        NewPop = NewPop(SortOrder);
% Now we must evalute each memebr of the NewPop
for i = 1:NumPop
% If NewPop is better than Pop
% (COMPLETE THIS: Write your code here)
if NewPop(i).TourLength <= Pop(i).TourLength
    % Candidate is better (shorter tour): accept it unconditionally,
    % replacing population member i with this candidate
    Pop(i) = NewPop(i);
% If NewPop is not better than Pop
% (COMPLETE THIS: Write your code here)
else
% Calculate Probability
% (COMPLETE THIS: Write your code here)
    % Candidate is worse: compute the relative increase in tour
    % length (DELTA), then the Boltzmann acceptance probability P,
    % which decreases as the candidate gets relatively worse or as
    % temperature T cools
    DELTA = (NewPop(i).TourLength - Pop(i).TourLength)/Pop(i).TourLength;
    P = exp(-DELTA/T);
    % Accept the worse candidate anyway with probability P,
    % allowing the algorithm to escape local optima
    if rand <= P
        Pop(i) = NewPop(i);
    end
end
% Update Best Solution Ever
% (COMPLETE THIS: Write your code here)
    % If population member i (after the accept/reject step above)
    % is better than the best solution found so far, record it as
    % the new global best
    if Pop(i).TourLength <= BestSolution.TourLength
        BestSolution = Pop(i);
    end
end
end
% Store Best Cost Ever
    BestTourLength(iteration) = BestSolution.TourLength;
% Update Temperature
    T = alpha * T;
% Display Iteration
    disp(['Iteration ' num2str(iteration) ...
': Best Cost = ' num2str(BestTourLength(iteration))]);
% Plot Best Tour
    figure(1)
    PlotSolution(BestSolution, Model);
end
%% Plot Obj. Func. Value vs Iteration
figure (2)
plot(BestTourLength,'LineWidth',2);
xlabel('Iteration Number');
ylabel('Best Tour Length');