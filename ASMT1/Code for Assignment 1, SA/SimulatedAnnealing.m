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
NumberCities = 50;
Model = CreateModel('Model 2', NumberCities);    

% Cost Function
ObjectiveFunction = @(tour) TourLength(tour, Model);    

%% Simulate Annealing Parameters
% Maximum Number of Iterations
MaxIter = 1000;  

% Maximum Number of Sub-iterations
MaxSubIter = 10; 

% Initial Temperature
T = 10;      

% Temperature Reduction Rate
alpha = 0.99;     

%% Simulate Annealing Initialization
% Create an Initial Solution
Solution.Tour = CreateRandomTour(Model);

% Evaluate the Initial Solution
Solution.TourLength = ObjectiveFunction(Solution.Tour);

% Initialize Best Solution Ever Found
BestSolution = Solution;

% Array to Hold Best TourLength Values
BestTourLength = zeros(MaxIter,1);

% Array to Hold Runtimes 
Runtime = zeros(MaxIter,1);

%% Simulate Annealing Main Loop
tic % Start Recording Runtime
for iteration = 1:MaxIter
    for subiteration = 1:MaxSubIter
        
        % Create a New Solution
        NewSolution.Tour = CreateNeighbour(Solution.Tour);
        
        % Evaluate the New Solution
        NewSolution.TourLength = ObjectiveFunction(NewSolution.Tour);
        
        % If NewSolution is better than Solution
        if NewSolution.TourLength <= Solution.TourLength 
            Solution = NewSolution;
        else % If NewSolution is not better than Solution
            DELTA = (NewSolution.TourLength - ...
                Solution.TourLength)/Solution.TourLength;
            P = exp(-DELTA/T); % Calculate Probability
            if rand <= P % Accepting or NOT Accepting 
                Solution = NewSolution;
            end
        end
        
        % Update Best Solution Ever
        if Solution.TourLength <= BestSolution.TourLength
            BestSolution = Solution;
        end
    end
    
    % Store Best TourLength Ever
    BestTourLength(iteration) = BestSolution.TourLength;
    
    % Update Temperature
    T = alpha * T;
    
    Runtime(iteration) = toc;
    
    % Display Iteration
    disp(['Iteration ' num2str(iteration) ...
        ': Best Cost = ' num2str(BestTourLength(iteration))]);
        
    % Plot Best Tour
    figure(1);
    PlotSolution(BestSolution, Model);
end

%% Plot Obj. Func. Value vs Iteration
figure (2)
subplot(211), plot(BestTourLength,'LineWidth',2);
xlabel('Iteration Number');
ylabel('Best Tour Length');

subplot(212), plot(Runtime, BestTourLength,'LineWidth',2);
xlabel('Runtime (secods)');
ylabel('Best Tour Length');



