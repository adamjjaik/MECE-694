%% Reference
% This Code was adopted from: 
% Mostapha Kalami Heris, Yarpiz Evolutionary Algorithms Toolbox for MATLAB (YPEA), Yarpiz, 2020.
% Link: https://www.mathworks.com/matlabcentral/fileexchange/71373-ypea?s_tid=prof_contriblnk
% This code was modifies for "MECE 788: Applied Computational Intelligence
% for Engineering" at the Univerisyt of Alberta, Winter 2021.

%% Initialization Step: Create a Nonlinear Function
clc
clear

[theta_t, ObjectiveFunction, numVar, VarMin, VarMax] = CreateProblem('Function 2');
VarSize = [1 numVar];

%% Default Genetic Algorithm Parameters
default_MaxIter     = 200;
default_NumPop      = 50;
default_pcrossover  = 0.6;
default_pmutation   = 0.4;

% Fixed settings (not being varied)
gamma = 0.05;
mu    = 0.02;
beta  = 8; % Selection pressure for Roulette Wheel Selection

num_reps = 25;

%% --- Parameter sets to test ---
MaxIter_values    = [50 100 200 400];
NumPop_values     = [10 25 50 100 200];
pcrossover_values = [0.3 0.6 0.9];
pmutation_values  = [0.2 0.4 0.6];

%% --- Investigate Maximum Iteration Number ---
mean_obj_iter = zeros(size(MaxIter_values));
std_obj_iter  = zeros(size(MaxIter_values));
mean_rt_iter  = zeros(size(MaxIter_values));
std_rt_iter   = zeros(size(MaxIter_values));

for p = 1:length(MaxIter_values)
    objs = zeros(num_reps,1);
    rts  = zeros(num_reps,1);
    for r = 1:num_reps
        [bestObj, rt] = RunGA(ObjectiveFunction, VarSize, VarMin, VarMax, ...
            MaxIter_values(p), default_NumPop, default_pcrossover, default_pmutation, gamma, mu, beta);
        objs(r) = bestObj;
        rts(r)  = rt;
    end
    mean_obj_iter(p) = mean(objs);
    std_obj_iter(p)  = std(objs);
    mean_rt_iter(p)  = mean(rts);
    std_rt_iter(p)   = std(rts);
    disp(['MaxIter = ' num2str(MaxIter_values(p)) ' done'])
end

%% --- Investigate Population Size ---
mean_obj_pop = zeros(size(NumPop_values));
std_obj_pop  = zeros(size(NumPop_values));
mean_rt_pop  = zeros(size(NumPop_values));
std_rt_pop   = zeros(size(NumPop_values));

for p = 1:length(NumPop_values)
    objs = zeros(num_reps,1);
    rts  = zeros(num_reps,1);
    for r = 1:num_reps
        [bestObj, rt] = RunGA(ObjectiveFunction, VarSize, VarMin, VarMax, ...
            default_MaxIter, NumPop_values(p), default_pcrossover, default_pmutation, gamma, mu, beta);
        objs(r) = bestObj;
        rts(r)  = rt;
    end
    mean_obj_pop(p) = mean(objs);
    std_obj_pop(p)  = std(objs);
    mean_rt_pop(p)  = mean(rts);
    std_rt_pop(p)   = std(rts);
    disp(['NumPop = ' num2str(NumPop_values(p)) ' done'])
end

%% --- Investigate Crossover Percentage ---
mean_obj_pc = zeros(size(pcrossover_values));
std_obj_pc  = zeros(size(pcrossover_values));
mean_rt_pc  = zeros(size(pcrossover_values));
std_rt_pc   = zeros(size(pcrossover_values));

for p = 1:length(pcrossover_values)
    objs = zeros(num_reps,1);
    rts  = zeros(num_reps,1);
    for r = 1:num_reps
        [bestObj, rt] = RunGA(ObjectiveFunction, VarSize, VarMin, VarMax, ...
            default_MaxIter, default_NumPop, pcrossover_values(p), default_pmutation, gamma, mu, beta);
        objs(r) = bestObj;
        rts(r)  = rt;
    end
    mean_obj_pc(p) = mean(objs);
    std_obj_pc(p)  = std(objs);
    mean_rt_pc(p)  = mean(rts);
    std_rt_pc(p)   = std(rts);
    disp(['pcrossover = ' num2str(pcrossover_values(p)) ' done'])
end

%% --- Investigate Mutation Percentage ---
mean_obj_pm = zeros(size(pmutation_values));
std_obj_pm  = zeros(size(pmutation_values));
mean_rt_pm  = zeros(size(pmutation_values));
std_rt_pm   = zeros(size(pmutation_values));

for p = 1:length(pmutation_values)
    objs = zeros(num_reps,1);
    rts  = zeros(num_reps,1);
    for r = 1:num_reps
        [bestObj, rt] = RunGA(ObjectiveFunction, VarSize, VarMin, VarMax, ...
            default_MaxIter, default_NumPop, default_pcrossover, pmutation_values(p), gamma, mu, beta);
        objs(r) = bestObj;
        rts(r)  = rt;
    end
    mean_obj_pm(p) = mean(objs);
    std_obj_pm(p)  = std(objs);
    mean_rt_pm(p)  = mean(rts);
    std_rt_pm(p)   = std(rts);
    disp(['pmutation = ' num2str(pmutation_values(p)) ' done'])
end

%% --- Plot results: one figure per parameter, Objective Value (left) and Runtime (right) ---

figure(1)
subplot(1,2,1)
errorbar(MaxIter_values, mean_obj_iter, std_obj_iter, '-ok', 'LineWidth', 2)
xlabel('Max Iterations'), ylabel('Objective Value'), title('Objective Value vs Max Iterations')
subplot(1,2,2)
errorbar(MaxIter_values, mean_rt_iter, std_rt_iter, '-ok', 'LineWidth', 2)
xlabel('Max Iterations'), ylabel('Runtime (s)'), title('Runtime vs Max Iterations')

figure(2)
subplot(1,2,1)
errorbar(NumPop_values, mean_obj_pop, std_obj_pop, '-ok', 'LineWidth', 2)
xlabel('Population Size'), ylabel('Objective Value'), title('Objective Value vs Population Size')
subplot(1,2,2)
errorbar(NumPop_values, mean_rt_pop, std_rt_pop, '-ok', 'LineWidth', 2)
xlabel('Population Size'), ylabel('Runtime (s)'), title('Runtime vs Population Size')

figure(3)
subplot(1,2,1)
errorbar(pcrossover_values, mean_obj_pc, std_obj_pc, '-ok', 'LineWidth', 2)
xlabel('Crossover Percentage'), ylabel('Objective Value'), title('Objective Value vs Crossover Percentage')
subplot(1,2,2)
errorbar(pcrossover_values, mean_rt_pc, std_rt_pc, '-ok', 'LineWidth', 2)
xlabel('Crossover Percentage'), ylabel('Runtime (s)'), title('Runtime vs Crossover Percentage')

figure(4)
subplot(1,2,1)
errorbar(pmutation_values, mean_obj_pm, std_obj_pm, '-ok', 'LineWidth', 2)
xlabel('Mutation Percentage'), ylabel('Objective Value'), title('Objective Value vs Mutation Percentage')
subplot(1,2,2)
errorbar(pmutation_values, mean_rt_pm, std_rt_pm, '-ok', 'LineWidth', 2)
xlabel('Mutation Percentage'), ylabel('Runtime (s)'), title('Runtime vs Mutation Percentage')

%% === Local Function: runs one full GA execution, returns final best obj value + runtime ===
function [bestObj, rt] = RunGA(ObjectiveFunction, VarSize, VarMin, VarMax, ...
    MaxIter, NumPop, pcrossover, pmutation, gamma, mu, beta)

    ncrossover = 2*round(pcrossover*NumPop/2);
    nmutation  = round(pmutation*NumPop);

    empty_solutions.Solution = [];
    empty_solutions.ObjValue = [];
    Pop = repmat(empty_solutions, NumPop, 1);

    for i = 1:NumPop
        Pop(i).Solution = unifrnd(VarMin, VarMax, VarSize);
        Pop(i).ObjValue = ObjectiveFunction(Pop(i).Solution);
    end

    ObjValues = [Pop.ObjValue];
    [ObjValues, SortOrder] = sort(ObjValues);
    Pop = Pop(SortOrder);

    BestSolution = Pop(1);
    WorstObjValue = Pop(end).ObjValue;

    tic
    for iteration = 1:MaxIter
        P = exp(-beta*ObjValues/WorstObjValue);
        P = P/sum(P);

        % Crossover (Roulette Wheel Selection, matching default script)
        popc = repmat(empty_solutions, ncrossover/2, 2);
        for i = 1:ncrossover/2
            i1 = RouletteWheelSelection(P);
            i2 = RouletteWheelSelection(P);
            p1 = Pop(i1);
            p2 = Pop(i2);
            [popc(i,1).Solution, popc(i,2).Solution] = ...
                ArithmeticCrossover(p1.Solution, p2.Solution, gamma, VarMin, VarMax);
            popc(i,1).ObjValue = ObjectiveFunction(popc(i,1).Solution);
            popc(i,2).ObjValue = ObjectiveFunction(popc(i,2).Solution);
        end
        popc = popc(:);

        % Mutation
        popm = repmat(empty_solutions,nmutation,1);
        for i = 1:nmutation
            i1 = randi([1 NumPop]);
            p = Pop(i1);
            popm(i).Solution = Mutation(p.Solution, mu, VarMin, VarMax);
            popm(i).ObjValue = ObjectiveFunction(popm(i).Solution);
        end

        Pop = [Pop; popc; popm]; %#ok
        ObjValues = [Pop.ObjValue];
        [ObjValues, SortOrder] = sort(ObjValues);
        Pop = Pop(SortOrder);

        WorstObjValue = max(WorstObjValue, Pop(end).ObjValue);

        Pop = Pop(1:NumPop);
        ObjValues = ObjValues(1:NumPop);

        BestSolution = Pop(1);
    end
    rt = toc;
    bestObj = BestSolution.ObjValue;
end