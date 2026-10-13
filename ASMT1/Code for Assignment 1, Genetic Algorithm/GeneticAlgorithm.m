%% Reference
% This Code was adopted from: 
% Mostapha Kalami Heris, Yarpiz Evolutionary Algorithms Toolbox for MATLAB (YPEA), Yarpiz, 2020.
% Link: https://www.mathworks.com/matlabcentral/fileexchange/71373-ypea?s_tid=prof_contriblnk

% This code was modifies for "MECE 788: Applied Computational Intelligence
% for Engineering" at the Univerisyt of Alberta, Winter 2021.

%% Initialization Step: Create a Nonlinear Function
clc
clear

% Nonlinear Function
[theta_t, ObjectiveFunction, numVar, VarMin, VarMax] = CreateProblem('Function 2');

% Decision Variables Matrix Size
VarSize = [1 numVar];

%% Genetic Algorithm Parameters
% Maximum Number of Iterations
MaxIter = 100;

% Population Size
NumPop = 50;

% Crossover Percentage
pcrossover = 0.6;
% Number of Offspring
ncrossover = 2*round(pcrossover*NumPop/2);
% Arithmetic Crossover Parameters
gamma = 0.05;

% Mutation Percentage
pmutation = 0.4;
% Number of Mutants
nmutation = round(pmutation*NumPop);
% Mutation Rate
mu = 0.02;

SelectionMethod = 'RWS';
switch SelectionMethod
    case 'RWS'
        UseRouletteWheelSelection = 1;
        UseTournamentSelection = 0;
        UseRandomSelection = 0;
        % Selection Pressure for Roulette Wheel Selection
        beta = 8;
    case 'TS'
        UseRouletteWheelSelection = 0;
        UseTournamentSelection = 1;
        UseRandomSelection = 0;
        % Tournamnet Size
        TournamentSize = 3;
    case 'RS'
        UseRouletteWheelSelection = 0;
        UseTournamentSelection = 0;
        UseRandomSelection = 1;
end

%% Initialize Genetic Algorithm
empty_solutions.Solution = [];
empty_solutions.ObjValue = [];
Pop = repmat(empty_solutions, NumPop, 1);

% Create Random Initial Population
for i = 1:NumPop
    % Initialize a Random Solution
    Pop(i).Solution = unifrnd(VarMin, VarMax, VarSize);
    % Evaluate the Random Solution
    Pop(i).ObjValue = ObjectiveFunction(Pop(i).Solution);
end

% Sort Initial Population
ObjValues = [Pop.ObjValue];
[ObjValues, SortOrder] = sort(ObjValues);
Pop = Pop(SortOrder);

% Record the Best Solution
BestSolution = Pop(1);

% Array to Hold Best ObjValues
BestObjValues = zeros(MaxIter, 1);

% Record Worst ObjValue
WorstObjValue = Pop(end).ObjValue; % end gives the last element

% Array to Hold Runtimes
Runtime = zeros(MaxIter, 1);

%% Genetic Algorithm Main Loop
tic
for iteration = 1:MaxIter
    % Calculate Selection Probabilities
    P = exp(-beta*ObjValues/WorstObjValue);
    P = P/sum(P);
    
    %%%% Apply Crossover Operator
    popc = repmat(empty_solutions, ncrossover/2, 2);
    for i = 1:ncrossover/2
        % Select Parents Indices
        if UseRouletteWheelSelection
            i1 = RouletteWheelSelection(P);
            i2 = RouletteWheelSelection(P);
        end
        if UseTournamentSelection
            i1 = TournamentSelection(Pop, TournamentSize);
            i2 = TournamentSelection(Pop, TournamentSize);
        end
        if UseRandomSelection
            i1 = randi([1 NumPop]);
            i2 = randi([1 NumPop]);
        end
        
        % Extract Parents from Population
        p1 = Pop(i1);
        p2 = Pop(i2);
        
        % Apply Arithmetic Crossover to Create Offspring
        [popc(i,1).Solution, popc(i,2).Solution] = ...
            ArithmeticCrossover(p1.Solution, p2.Solution, gamma, VarMin, VarMax);
        
        % Evaluate Offspring
        popc(i,1).ObjValue = ObjectiveFunction(popc(i,1).Solution);
        popc(i,2).ObjValue = ObjectiveFunction(popc(i,2).Solution);
    end
    popc = popc(:); 
    
    %%%% Apply Mutation Operator
    popm = repmat(empty_solutions,nmutation,1);
    for i = 1:nmutation
        % Select Parent Randomly
        i1 = randi([1 NumPop]);
        p = Pop(i1);
        
        % Apply Mutation to Create Offspring
        popm(i).Solution = Mutation(p.Solution, mu, VarMin, VarMax);
        % Evaluate Offspring
        popm(i).ObjValue = ObjectiveFunction(popm(i).Solution);
    end
    
    % Create Merged Population
    Pop = [Pop; popc; popm]; %#ok
    
    % Sort Population
    ObjValues = [Pop.ObjValue]; % Extract Objective Values
    [ObjValues, SortOrder] = sort(ObjValues);
    Pop = Pop(SortOrder);
    
    % Update Worst ObjValue
    WorstObjValue = max(WorstObjValue, Pop(end).ObjValue);
    
    % Truncation
    Pop = Pop(1:NumPop);
    ObjValues = ObjValues(1:NumPop);
    
    % Record Best Solution Ever 
    BestSolution = Pop(1);
    
    % Store Best ObjValue Ever 
    BestObjValues(iteration) = BestSolution.ObjValue;
    
    % Store Runtime
    Runtime(iteration) = toc;
    
    % Show Iteration Information
    disp(['Iteration ' num2str(iteration) ...
        ', Best Objective Value = ' num2str(BestObjValues(iteration))]);
end

%% Results
figure(1)
plot(Runtime, BestObjValues,'LineWidth',2);
xlabel('Runtime (seconds)');
ylabel('Objective Value');

% Compare Parameter Values
disp(['Expected Theta values: ' ...
    'Theta0 = ' num2str(round(theta_t(1),2)) ', Theta1 = ' num2str(round(theta_t(2),2)) ...
    ', Theta3 = ' num2str(round(theta_t(3),2)) ', Theta4 = ' num2str(round(theta_t(4),2))]);

disp(['Theta found by GA:     ' ...
    'Theta0 = ' num2str(round(BestSolution.Solution(1),2)) ...
    ', Theta1 = ' num2str(round(BestSolution.Solution(2),2)) ...
    ', Theta3 = ' num2str(round(BestSolution.Solution(3),2)) ...
    ', Theta4 = ' num2str(round(BestSolution.Solution(4),2))]);





