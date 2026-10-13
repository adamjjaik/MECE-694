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

%% Genetic Algorithm Default Parameters (from Question 1)
MaxIter    = 200;
NumPop     = 50;
pcrossover = 0.6;
pmutation  = 0.4;
gamma      = 0.05;   % Arithmetic Crossover parameter
alpha_blx  = 0.3;    % BLX-alpha parameter (NewCrossover), per literature-recommended value
mu         = 0.02;
beta       = 8;      % Selection pressure for Roulette Wheel Selection

num_reps = 25;

%% --- Run GA 25 times for each crossover method ---
methods = {'Arithmetic', 'Heuristic', 'New'};
mean_obj = zeros(1,3);
std_obj  = zeros(1,3);
mean_rt  = zeros(1,3);
std_rt   = zeros(1,3);

for m = 1:length(methods)
    objs = zeros(num_reps,1);
    rts  = zeros(num_reps,1);
    for r = 1:num_reps
        [bestObj, rt] = RunGA(ObjectiveFunction, VarSize, VarMin, VarMax, ...
            MaxIter, NumPop, pcrossover, pmutation, gamma, alpha_blx, mu, beta, methods{m});
        objs(r) = bestObj;
        rts(r)  = rt;
    end
    mean_obj(m) = mean(objs);
    std_obj(m)  = std(objs);
    mean_rt(m)  = mean(rts);
    std_rt(m)   = std(rts);
    disp([methods{m} ' Crossover done'])
end

%% --- Summary Table ---
disp('Method       Mean Obj      Std Obj       Mean Runtime   Std Runtime')
for m = 1:3
    disp([methods{m} '  ' num2str(mean_obj(m)) '  ' num2str(std_obj(m)) ...
        '  ' num2str(mean_rt(m)) '  ' num2str(std_rt(m))])
end

%% === Local Function: runs one full GA execution with a chosen crossover method ===
function [bestObj, rt] = RunGA(ObjectiveFunction, VarSize, VarMin, VarMax, ...
    MaxIter, NumPop, pcrossover, pmutation, gamma, alpha_blx, mu, beta, method)

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

        popc = repmat(empty_solutions, ncrossover/2, 2);
        for i = 1:ncrossover/2
            switch method
                case 'Arithmetic'
                    i1 = RouletteWheelSelection(P);
                    i2 = RouletteWheelSelection(P);
                    p1 = Pop(i1);
                    p2 = Pop(i2);
                    [popc(i,1).Solution, popc(i,2).Solution] = ...
                        ArithmeticCrossover(p1.Solution, p2.Solution, gamma, VarMin, VarMax);

                case 'Heuristic'
                    i1 = RouletteWheelSelection(P);
                    i2 = RouletteWheelSelection(P);
                    i3 = RouletteWheelSelection(P);
                    i4 = RouletteWheelSelection(P);
                    p1 = Pop(i1); p2 = Pop(i2); p3 = Pop(i3); p4 = Pop(i4);
                    [popc(i,1).Solution, popc(i,2).Solution] = ...
                        HeuristicCrossover(p1, p2, p3, p4, VarMin, VarMax);

                case 'New'
                    i1 = RouletteWheelSelection(P);
                    i2 = RouletteWheelSelection(P);
                    p1 = Pop(i1);
                    p2 = Pop(i2);
                    [popc(i,1).Solution, popc(i,2).Solution] = ...
                        NewCrossover(p1.Solution, p2.Solution, alpha_blx, VarMin, VarMax);
            end

            popc(i,1).ObjValue = ObjectiveFunction(popc(i,1).Solution);
            popc(i,2).ObjValue = ObjectiveFunction(popc(i,2).Solution);
        end
        popc = popc(:);

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