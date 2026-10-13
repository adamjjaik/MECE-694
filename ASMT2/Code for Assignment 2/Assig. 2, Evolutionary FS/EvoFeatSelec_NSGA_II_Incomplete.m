%% Reference
% This Code was adopted from: 
% Mostapha Kalami Heris, NSGA-III: Non-dominated Sorting Genetic Algorithm, 
% the Third Version — MATLAB Implementation (URL: https://yarpiz.com/456/ypea126-nsga3), Yarpiz, 2016.
% Link: https://www.mathworks.com/matlabcentral/fileexchange/60678-nsga-iii-in-matlab?s_tid=prof_contriblnk

% This code was modifies for "MECE 788: Applied Computational Intelligence
% for Engineering" at the Univerisyt of Alberta, Winter 2021.

%% Initialization and Problem Creation
clc
clear

% BodyFat or BreastCancer
Dataset_Name = 'BreastCancer';

% Load Dataset
Data = LoadData(Dataset_Name);

% Number of Decision Variables
NumVar = Data.NumFeatures;

% Decision Variables Matrix Size
VarSize = [1 NumVar];

% Number of Objective Functions
NumObjective = 2;

% Weight for Error in Training Data (from 0 to 1)
WeightTrain = 0.5;
% Number of Artificial Neural Network Evaluations
NumExecutions = 10;

% Create Objectve Function
Objectives = @(SelectedFeatures) FeatureSelectionObjective(...
    SelectedFeatures, Data, WeightTrain, NumExecutions);

%% Define NSGA-II Parameters
% Maximum Number of Iterations
MaxIter = 15;

% Population Size
NumPop = 10;

% Crossover Percentage
pcrossover = 0.7;
% Number of Parnets (Offspring)
ncrossover = 2*round(pcrossover*NumPop/2);

% Mutation Percentage
pmutation = 0.4;
% Number of Mutants
nmutation = round(pmutation*NumPop);
% Mutation Rate
mu = 0.02;

%% Initialize NSGA-II
empty_solutions.Solution = [];
empty_solutions.ObjValue = [];
empty_solutions.ExtraInfo = [];
empty_solutions.Rank = [];
empty_solutions.DominationSet = [];
empty_solutions.DominatedCount = [];
empty_solutions.CrowdingDistance = [];
Pop = repmat(empty_solutions,NumPop,1);

% Create Random Initial Population
for i = 1:NumPop
    % Initialize a Random Solution
    if i == 1 % Include All Features
        Pop(i).Solution = ones(VarSize);
    else
        Pop(i).Solution = randi([0 1],VarSize);
    end
    % Evaluate the Random Solution
    [Pop(i).ObjValue, Pop(i).ExtraInfo] = Objectives(Pop(i).Solution);
end

% Non-Dominated Sorting of Initial Solution
[Pop, F] = NonDominatedSorting(Pop);

% Calculating Crowding Distance
Pop = CalculateCrowdingDistance(Pop,F);

% Sorting Population
[Pop, F] = SortPopulation(Pop);

%% NSGA-II Main Loop
tic
for iteration = 1:MaxIter
    %%%% Apply Crossover Operator
    Popc = repmat(empty_solutions, ncrossover/2, 2);
    for i = 1:ncrossover/2
        % Select Parent Randomly
        i1 = randi([1 NumPop]);
        i2 = randi([1 NumPop]);
        % Extract Parents from Population
        p1 = Pop(i1);
        p2 = Pop(i2);
        % Apply Crossover to Create Offspring (!!! Complete This Section)
        [Popc(i,1).Solution, Popc(i,2).Solution] = ...
            BinaryCrossover_Incomplete(p1.Solution, p2.Solution);
        % Evaluate Offspring
        [Popc(i,1).ObjValue, Popc(i,1).ExtraInfo] = Objectives(Popc(i,1).Solution);
        [Popc(i,2).ObjValue, Popc(i,2).ExtraInfo] = Objectives(Popc(i,2).Solution);
    end
    Popc = Popc(:);
    
    %%%% Apply Mutation Operator
    Popm = repmat(empty_solutions, nmutation, 1);
    for i = 1:nmutation
        % Select Parent Randomly
        i1 = randi([1 NumPop]);
        p = Pop(i1);
        % Apply Mutation to Create Offspring (!!! Complete This Section)
        Popm(i).Solution = Mutation_Incomplete(p.Solution, mu);
        % Evaluate Offspring
        [Popm(i).ObjValue, Popm(i).ExtraInfo] = Objectives(Popm(i).Solution);
    end
    
    % Create Merged Population
    Pop = [Pop; Popc; Popm];
    
    % Perform Non-Dominated Sorting: Before Truncation
    [Pop, F] = NonDominatedSorting(Pop);
    % Calculate Crowding Distance: Before Truncation
    Pop = CalculateCrowdingDistance(Pop, F);
    % Sorting Population: Before Truncation
    [Pop, ~] = SortPopulation(Pop);
    
    % Truncate
    Pop = Pop(1:NumPop);
    
    % Non-Dominated Sorting: After Truncation
    [Pop, F]=NonDominatedSorting(Pop);
    % Calculate Crowding Distance: After Truncation
    Pop = CalculateCrowdingDistance(Pop,F);
    % Sort Population: After Truncation
    [Pop, F]=SortPopulation(Pop);
    
    % Record Pareto Front F1
    F1 = Pop(F{1});
    F1 = FindUniqueSolutions(F1);
    
    % Show Iteration Information
    disp(['Iteration ' num2str(iteration) ' From ' num2str(MaxIter)]);
    
    % Plot F1 Costs
    figure(1);
    PlotObjectives(F1);
end


