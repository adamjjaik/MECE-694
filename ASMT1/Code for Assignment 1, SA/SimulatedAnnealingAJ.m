%% Reference
% This Code was adopted from:
% Mostapha Kalami Heris, Yarpiz Evolutionary Algorithms Toolbox for MATLAB (YPEA), Yarpiz, 2020.
% Link: https://www.mathworks.com/matlabcentral/fileexchange/71373-ypea?s_tid=prof_contriblnk
% This code was modifies for "MECE 788: Applied Computational Intelligence
% for Engineering" at the Univerisyt of Alberta, Winter 2021.

%% Initialization: Problem Definition
clc
clear

% Create Problem Model (Model 2, fixed for this investigation)
NumberCities = 30;
Model = CreateModel('Model 2', NumberCities);
ObjectiveFunction = @(tour) TourLength(tour, Model);

%% Default Simulated Annealing Parameters
default_MaxIter    = 800;
default_MaxSubIter = 10;
default_T          = 10;
default_alpha      = 0.99;

num_reps = 25;

%% --- Parameter sets to test ---
MaxIter_values    = [200 400 600 800 1000 1200];
MaxSubIter_values = [5 10 15];
T_values          = [1 5 10 15 20];
alpha_values      = [0.90 0.95 0.99 0.999];

%% --- Investigate Maximum Iteration Number ---
mean_len_iter = zeros(size(MaxIter_values));
std_len_iter  = zeros(size(MaxIter_values));
mean_rt_iter  = zeros(size(MaxIter_values));
std_rt_iter   = zeros(size(MaxIter_values));

for p = 1:length(MaxIter_values)
    lens = zeros(num_reps,1);
    rts  = zeros(num_reps,1);
    for r = 1:num_reps
        [bestLen, rt] = RunSA(Model, ObjectiveFunction, ...
            MaxIter_values(p), default_MaxSubIter, default_T, default_alpha);
        lens(r) = bestLen;
        rts(r)  = rt;
    end
    mean_len_iter(p) = mean(lens);
    std_len_iter(p)  = std(lens);
    mean_rt_iter(p)  = mean(rts);
    std_rt_iter(p)   = std(rts);
    disp(['MaxIter = ' num2str(MaxIter_values(p)) ' done'])
end

%% --- Investigate Maximum Sub-Iteration Number ---
mean_len_sub = zeros(size(MaxSubIter_values));
std_len_sub  = zeros(size(MaxSubIter_values));
mean_rt_sub  = zeros(size(MaxSubIter_values));
std_rt_sub   = zeros(size(MaxSubIter_values));

for p = 1:length(MaxSubIter_values)
    lens = zeros(num_reps,1);
    rts  = zeros(num_reps,1);
    for r = 1:num_reps
        [bestLen, rt] = RunSA(Model, ObjectiveFunction, ...
            default_MaxIter, MaxSubIter_values(p), default_T, default_alpha);
        lens(r) = bestLen;
        rts(r)  = rt;
    end
    mean_len_sub(p) = mean(lens);
    std_len_sub(p)  = std(lens);
    mean_rt_sub(p)  = mean(rts);
    std_rt_sub(p)   = std(rts);
    disp(['MaxSubIter = ' num2str(MaxSubIter_values(p)) ' done'])
end

%% --- Investigate Initial Temperature ---
mean_len_T = zeros(size(T_values));
std_len_T  = zeros(size(T_values));
mean_rt_T  = zeros(size(T_values));
std_rt_T   = zeros(size(T_values));

for p = 1:length(T_values)
    lens = zeros(num_reps,1);
    rts  = zeros(num_reps,1);
    for r = 1:num_reps
        [bestLen, rt] = RunSA(Model, ObjectiveFunction, ...
            default_MaxIter, default_MaxSubIter, T_values(p), default_alpha);
        lens(r) = bestLen;
        rts(r)  = rt;
    end
    mean_len_T(p) = mean(lens);
    std_len_T(p)  = std(lens);
    mean_rt_T(p)  = mean(rts);
    std_rt_T(p)   = std(rts);
    disp(['T = ' num2str(T_values(p)) ' done'])
end

%% --- Investigate Temperature Reduction Rate ---
mean_len_alpha = zeros(size(alpha_values));
std_len_alpha  = zeros(size(alpha_values));
mean_rt_alpha  = zeros(size(alpha_values));
std_rt_alpha   = zeros(size(alpha_values));

for p = 1:length(alpha_values)
    lens = zeros(num_reps,1);
    rts  = zeros(num_reps,1);
    for r = 1:num_reps
        [bestLen, rt] = RunSA(Model, ObjectiveFunction, ...
            default_MaxIter, default_MaxSubIter, default_T, alpha_values(p));
        lens(r) = bestLen;
        rts(r)  = rt;
    end
    mean_len_alpha(p) = mean(lens);
    std_len_alpha(p)  = std(lens);
    mean_rt_alpha(p)  = mean(rts);
    std_rt_alpha(p)   = std(rts);
    disp(['alpha = ' num2str(alpha_values(p)) ' done'])
end

%% --- Plot results: one figure per parameter, Tour Length (left) and Runtime (right) ---

figure(1)
subplot(1,2,1)
errorbar(MaxIter_values, mean_len_iter, std_len_iter, '-ok', 'LineWidth', 2)
xlabel('Max Iterations'), ylabel('Tour Length'), title('Tour Length vs Max Iterations')
subplot(1,2,2)
errorbar(MaxIter_values, mean_rt_iter, std_rt_iter, '-ok', 'LineWidth', 2)
xlabel('Max Iterations'), ylabel('Runtime (s)'), title('Runtime vs Max Iterations')

figure(2)
subplot(1,2,1)
errorbar(MaxSubIter_values, mean_len_sub, std_len_sub, '-ok', 'LineWidth', 2)
xlabel('Max Sub-Iterations'), ylabel('Tour Length'), title('Tour Length vs Max Sub-Iterations')
subplot(1,2,2)
errorbar(MaxSubIter_values, mean_rt_sub, std_rt_sub, '-ok', 'LineWidth', 2)
xlabel('Max Sub-Iterations'), ylabel('Runtime (s)'), title('Runtime vs Max Sub-Iterations')

figure(3)
subplot(1,2,1)
errorbar(T_values, mean_len_T, std_len_T, '-ok', 'LineWidth', 2)
xlabel('Initial Temperature'), ylabel('Tour Length'), title('Tour Length vs Initial Temperature')
subplot(1,2,2)
errorbar(T_values, mean_rt_T, std_rt_T, '-ok', 'LineWidth', 2)
xlabel('Initial Temperature'), ylabel('Runtime (s)'), title('Runtime vs Initial Temperature')

figure(4)
subplot(1,2,1)
errorbar(alpha_values, mean_len_alpha, std_len_alpha, '-ok', 'LineWidth', 2)
xlabel('Temperature Reduction Rate'), ylabel('Tour Length'), title('Tour Length vs Alpha')
subplot(1,2,2)
errorbar(alpha_values, mean_rt_alpha, std_rt_alpha, '-ok', 'LineWidth', 2)
xlabel('Temperature Reduction Rate'), ylabel('Runtime (s)'), title('Runtime vs Alpha')

%% === Local Function: runs one full SA execution, returns final best length + total runtime ===
function [bestLen, rt] = RunSA(Model, ObjectiveFunction, MaxIter, MaxSubIter, T0, alpha)
    Solution.Tour = CreateRandomTour(Model);
    Solution.TourLength = ObjectiveFunction(Solution.Tour);
    BestSolution = Solution;
    T = T0;

    tic
    for iteration = 1:MaxIter
        for subiteration = 1:MaxSubIter
            NewSolution.Tour = CreateNeighbour(Solution.Tour);
            NewSolution.TourLength = ObjectiveFunction(NewSolution.Tour);
            if NewSolution.TourLength <= Solution.TourLength
                Solution = NewSolution;
            else
                DELTA = (NewSolution.TourLength - Solution.TourLength)/Solution.TourLength;
                P = exp(-DELTA/T);
                if rand <= P
                    Solution = NewSolution;
                end
            end
            if Solution.TourLength <= BestSolution.TourLength
                BestSolution = Solution;
            end
        end
        T = alpha * T;
    end
    rt = toc;
    bestLen = BestSolution.TourLength;
end