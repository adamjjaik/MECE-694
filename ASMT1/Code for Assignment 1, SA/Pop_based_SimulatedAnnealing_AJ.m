%% Reference
% This Code was adopted from:
% Mostapha Kalami Heris, Yarpiz Evolutionary Algorithms Toolbox for MATLAB (YPEA), Yarpiz, 2020.
%% Initialization: Problem Definition
clc
clear

NumberCities = 30;
Model = CreateModel('Model 2', NumberCities);
ObjectiveFunction = @(tour) TourLength(tour, Model);

% Best Parameters from Question 1
MaxIter    = 1000;
MaxSubIter = 15;
T0         = 1;
alpha      = 0.90;

% Population-based SA specific parameters
NumPop   = 3;
NumNeigh = 5;

num_reps = 25;

%% --- Run Simple SA 25 times ---
simple_lens = zeros(num_reps,1);
simple_rts  = zeros(num_reps,1);
for r = 1:num_reps
    [bestLen, rt] = RunSimpleSA(Model, ObjectiveFunction, MaxIter, MaxSubIter, T0, alpha);
    simple_lens(r) = bestLen;
    simple_rts(r)  = rt;
end

%% --- Run Population-based SA 25 times ---
pop_lens = zeros(num_reps,1);
pop_rts  = zeros(num_reps,1);
for r = 1:num_reps
    [bestLen, rt] = RunPopSA(Model, ObjectiveFunction, MaxIter, MaxSubIter, T0, alpha, NumPop, NumNeigh);
    pop_lens(r) = bestLen;
    pop_rts(r)  = rt;
end

%% --- Summary statistics ---
mean_len = [mean(simple_lens), mean(pop_lens)];
std_len  = [std(simple_lens), std(pop_lens)];
mean_rt  = [mean(simple_rts), mean(pop_rts)];
std_rt   = [std(simple_rts), std(pop_rts)];

disp('Simple SA  - Mean Tour Length / Std / Mean Runtime / Std:')
disp([mean_len(1), std_len(1), mean_rt(1), std_rt(1)])
disp('Pop-based SA - Mean Tour Length / Std / Mean Runtime / Std:')
disp([mean_len(2), std_len(2), mean_rt(2), std_rt(2)])

%% --- Plot: Tour Length (left) and Runtime (right), Simple vs Pop-based ---
figure(1)
subplot(1,2,1)
bar(mean_len, 'FaceColor', [0.7 0.7 0.7])
hold on
errorbar(1:2, mean_len, std_len, 'k', 'LineStyle', 'none', 'LineWidth', 2)
set(gca, 'XTickLabel', {'Simple SA','Pop-based SA'})
ylabel('Tour Length')
title('Tour Length Comparison')
hold off

subplot(1,2,2)
bar(mean_rt, 'FaceColor', [0.7 0.7 0.7])
hold on
errorbar(1:2, mean_rt, std_rt, 'k', 'LineStyle', 'none', 'LineWidth', 2)
set(gca, 'XTickLabel', {'Simple SA','Pop-based SA'})
ylabel('Runtime (s)')
title('Runtime Comparison')
hold off

%% === Local Function: Simple SA, returns final best length + runtime ===
function [bestLen, rt] = RunSimpleSA(Model, ObjectiveFunction, MaxIter, MaxSubIter, T0, alpha)
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

%% === Local Function: Population-based SA, returns final best length + runtime ===
function [bestLen, rt] = RunPopSA(Model, ObjectiveFunction, MaxIter, MaxSubIter, T0, alpha, NumPop, NumNeigh)
    empty_solutions.Tour = [];
    empty_solutions.TourLength = [];
    Pop = repmat(empty_solutions, NumPop, 1);
    BestSolution.TourLength = inf;

    for i = 1:NumPop
        Pop(i).Tour = CreateRandomTour(Model);
        Pop(i).TourLength = ObjectiveFunction(Pop(i).Tour);
        if Pop(i).TourLength <= BestSolution.TourLength
            BestSolution = Pop(i);
        end
    end

    T = T0;
    tic
    for iteration = 1:MaxIter
        for subiteration = 1:MaxSubIter
            NewPop = repmat(empty_solutions, NumPop, NumNeigh);
            for i = 1:NumPop
                for j = 1:NumNeigh
                    NewPop(i,j).Tour = CreateNeighbour(Pop(i).Tour);
                    NewPop(i,j).TourLength = ObjectiveFunction(NewPop(i,j).Tour);
                end
            end
            NewPop = NewPop(:);
            [~, SortOrder] = sort([NewPop.TourLength]);
            NewPop = NewPop(SortOrder);

            for i = 1:NumPop
                if NewPop(i).TourLength <= Pop(i).TourLength
                    Pop(i) = NewPop(i);
                else
                    DELTA = (NewPop(i).TourLength - Pop(i).TourLength)/Pop(i).TourLength;
                    P = exp(-DELTA/T);
                    if rand <= P
                        Pop(i) = NewPop(i);
                    end
                end
                if Pop(i).TourLength <= BestSolution.TourLength
                    BestSolution = Pop(i);
                end
            end
        end
        T = alpha * T;
    end
    rt = toc;
    bestLen = BestSolution.TourLength;
end