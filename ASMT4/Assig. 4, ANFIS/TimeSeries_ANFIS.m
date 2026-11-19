%% TimeSeries_ANFIS.m
% MECE 694 - Assignment 4, Section 2: Time-series prediction with ANFIS
% Questions 2, 3 and 4
% Requires: TimeSeries_Data.m (same folder), TimeSeriesData.mat,
%           Fuzzy Logic Toolbox, Deep Learning Toolbox (for dividerand)
clear; clc; close all;

%% Q2 - Create input/output data
load TimeSeriesData.mat              % loads x: 1 x 1201 Mackey-Glass series
delays = [6 12 18 24];
[Inputs, Targets] = TimeSeries_Data(x, delays);
fprintf('Dataset: %d samples x %d inputs (delays = [%s])\n', ...
    size(Inputs,1), size(Inputs,2), num2str(delays));

%% Q3 - Random train / validation / test split
Seed   = 1;                          % fixed seed -> repeatable split & results
Ratios = [0.70 0.15 0.15];           % train / validation / test
Data   = SplitData(Inputs, Targets, Ratios, Seed);
fprintf('Train: %d | Validation: %d | Test: %d\n\n', ...
    numel(Data.Y_train), numel(Data.Y_val), numel(Data.Y_test));

%% Q4 - ANFIS settings (kept identical for every run so comparisons are fair)
ANFIS.EpochNumber          = 100;    % max training epochs
ANFIS.InitialStepSize      = 0.01;   % gradient step size for premise params
ANFIS.StepSizeDecreaseRate = 0.9;    % shrink step if error oscillates
ANFIS.StepSizeIncreaseRate = 1.1;    % grow step if error keeps decreasing
ANFIS.OptimizationMethod   = 1;      % 1 = hybrid (least squares + backprop)

%% Q4 - Parameter sweep: three raw-FIS methods x three parameter values
% Only ONE parameter is changed per method; everything else stays fixed.
%   GridPartition         -> NumMembershipFunctions per input (gaussmf)
%   SubtractiveClustering -> ClusterInfluenceRange (radius)
%   FCMClustering         -> NumClusters
Runs = {
    'GridPartition',          2      % 2^4 = 16 rules
    'GridPartition',          3      % 3^4 = 81 rules
    'GridPartition',          4      % 4^4 = 256 rules (slowest, may overfit)
    'SubtractiveClustering',  0.3
    'SubtractiveClustering',  0.5
    'SubtractiveClustering',  0.7
    'FCMClustering',          4
    'FCMClustering',          8
    'FCMClustering',          12
    };

nRuns      = size(Runs, 1);
Method     = strings(nRuns, 1);
Param      = zeros(nRuns, 1);
NumRules   = zeros(nRuns, 1);
RMSE_Train = zeros(nRuns, 1);
RMSE_Val   = zeros(nRuns, 1);
RMSE_Test  = zeros(nRuns, 1);
Time_s     = zeros(nRuns, 1);
FISs       = cell(nRuns, 1);
Hist       = cell(nRuns, 1);

for r = 1:nRuns
    Method(r) = string(Runs{r,1});
    Param(r)  = Runs{r,2};

    [FISs{r}, Res] = RunANFIS(Method(r), Param(r), Data, ANFIS, Seed);

    NumRules(r)   = Res.NumRules;
    RMSE_Train(r) = Res.RMSE_Train;
    RMSE_Val(r)   = Res.RMSE_Val;
    RMSE_Test(r)  = Res.RMSE_Test;
    Time_s(r)     = Res.Time;
    Hist{r}       = Res;

    fprintf('%-22s param = %-4g rules = %-4d RMSE train = %.5f | val = %.5f | test = %.5f  (%.1f s)\n', ...
        Method(r), Param(r), NumRules(r), RMSE_Train(r), RMSE_Val(r), RMSE_Test(r), Time_s(r));
end

Results = table(Method, Param, NumRules, RMSE_Train, RMSE_Val, RMSE_Test, Time_s);
disp(Results)
writetable(Results, 'Q4_RMSE_Results.csv');   % copy into your report table

%% Q4 - Plots for the best setting of each method
% "Best" is chosen by VALIDATION RMSE (the test set is never used to choose)
MethodNames = unique(Method, 'stable');
BestRow = zeros(numel(MethodNames), 1);
for m = 1:numel(MethodNames)
    rows = find(Method == MethodNames(m));
    [~, k] = min(RMSE_Val(rows));
    BestRow(m) = rows(k);
    b = BestRow(m);

    Label = sprintf('%s (param = %g, %d rules)', MethodNames(m), Param(b), NumRules(b));
    PlotPredictions(FISs{b}, Data, delays, Label);

    % Training vs validation error curves (shows where chkFIS was taken)
    figure('Name', ['Learning curve - ' char(Label)]);
    plot(Hist{b}.trainError, 'LineWidth', 1.5, 'DisplayName', 'Training RMSE'); hold on
    plot(Hist{b}.chkError,   'LineWidth', 1.5, 'DisplayName', 'Validation RMSE'); hold off
    xlabel('Epoch'), ylabel('RMSE'), legend, grid on
    title(['ANFIS learning curve: ' char(Label)], 'Interpreter', 'none')
end

%% Q4 - Best overall FIS settings with two other sets of delays
[~, bestOverall] = min(RMSE_Val);
fprintf('\nBest overall: %s, param = %g\n', Method(bestOverall), Param(bestOverall));

DelaySets = {delays, [3 6 9 12], [6 12 18]};   % first = original, for comparison
nD = numel(DelaySets);
DelayStr  = strings(nD,1);
D_Rules   = zeros(nD,1); D_Train = zeros(nD,1);
D_Val     = zeros(nD,1); D_Test  = zeros(nD,1);

for d = 1:nD
    [In_d, Out_d] = TimeSeries_Data(x, DelaySets{d});
    Data_d = SplitData(In_d, Out_d, Ratios, Seed);
    [FIS_d, Res_d] = RunANFIS(Method(bestOverall), Param(bestOverall), Data_d, ANFIS, Seed);

    DelayStr(d) = "[" + num2str(DelaySets{d}) + "]";
    D_Rules(d) = Res_d.NumRules;
    D_Train(d) = Res_d.RMSE_Train;
    D_Val(d)   = Res_d.RMSE_Val;
    D_Test(d)  = Res_d.RMSE_Test;

    if d > 1
        PlotPredictions(FIS_d, Data_d, DelaySets{d}, ...
            sprintf('%s, delays = %s', Method(bestOverall), DelayStr(d)));
    end
end

DelayResults = table(DelayStr, D_Rules, D_Train, D_Val, D_Test, ...
    'VariableNames', {'Delays', 'NumRules', 'RMSE_Train', 'RMSE_Val', 'RMSE_Test'});
disp(DelayResults)
writetable(DelayResults, 'Q4_Delay_Results.csv');


%% ======================= Local functions =======================
% (Local functions in a script must be at the end of the file.)

function Data = SplitData(Inputs, Targets, Ratios, seed)
% Randomly split data into train / validation / test sets.
    rng(seed);
    N = size(Inputs, 1);
    [Data.trainIdx, Data.valIdx, Data.testIdx] = ...
        dividerand(N, Ratios(1), Ratios(2), Ratios(3));
    Data.X_train = Inputs(Data.trainIdx, :);  Data.Y_train = Targets(Data.trainIdx);
    Data.X_val   = Inputs(Data.valIdx, :);    Data.Y_val   = Targets(Data.valIdx);
    Data.X_test  = Inputs(Data.testIdx, :);   Data.Y_test  = Targets(Data.testIdx);
end

function [bestFIS, Res] = RunANFIS(method, param, Data, A, seed)
% Build a raw Sugeno FIS with genfis, tune it with ANFIS, return the
% validation-best FIS (chkFIS) and its RMSE on train / val / test.
    rng(seed);   % FCM uses random initial centres -> keep repeatable

    %%% STEP 1: Raw FIS
    switch char(method)
        case 'GridPartition'
            opt = genfisOptions('GridPartition', ...
                'NumMembershipFunctions', param, ...
                'InputMembershipFunctionType', 'gaussmf', ...
                'OutputMembershipFunctionType', 'linear');
        case 'SubtractiveClustering'
            opt = genfisOptions('SubtractiveClustering', ...
                'ClusterInfluenceRange', param, ...   % scalar = same for all inputs
                'SquashFactor', 1.25, 'AcceptRatio', 0.5, 'RejectRatio', 0.15, ...
                'Verbose', false);
        case 'FCMClustering'
            opt = genfisOptions('FCMClustering', ...
                'FISType', 'sugeno', ...              % ANFIS needs Sugeno!
                'NumClusters', param, 'Exponent', 2, ...
                'MaxNumIteration', 100, 'Verbose', false);
        otherwise
            error('Unknown method: %s', method);
    end
    rawFIS = genfis(Data.X_train, Data.Y_train, opt);

    %%% STEP 2: ANFIS options
    aopt = anfisOptions('InitialFIS', rawFIS, ...
        'EpochNumber', A.EpochNumber, 'ErrorGoal', 0, ...
        'InitialStepSize', A.InitialStepSize, ...
        'StepSizeDecreaseRate', A.StepSizeDecreaseRate, ...
        'StepSizeIncreaseRate', A.StepSizeIncreaseRate, ...
        'ValidationData', [Data.X_val, Data.Y_val], ...
        'OptimizationMethod', A.OptimizationMethod, ...
        'DisplayANFISInformation', 0, 'DisplayErrorValues', 0, ...
        'DisplayStepSize', 0, 'DisplayFinalResults', 0);

    %%% STEP 3: Train. chkFIS = FIS at the epoch with lowest validation error
    tic
    [~, trainError, ~, bestFIS, chkError] = anfis([Data.X_train, Data.Y_train], aopt);
    Res.Time = toc;

    %%% STEP 4: Evaluate
    ev   = evalfisOptions('OutOfRangeInputValueMessage', 'none', ...
                          'NoRuleFiredMessage', 'none');
    rmse = @(y, yhat) sqrt(mean((y - yhat).^2));

    Res.NumRules   = numel(bestFIS.Rules);
    Res.RMSE_Train = rmse(Data.Y_train, evalfis(bestFIS, Data.X_train, ev));
    Res.RMSE_Val   = rmse(Data.Y_val,   evalfis(bestFIS, Data.X_val,   ev));
    Res.RMSE_Test  = rmse(Data.Y_test,  evalfis(bestFIS, Data.X_test,  ev));
    Res.trainError = trainError;
    Res.chkError   = chkError;
end

function PlotPredictions(FIS, Data, delays, Label)
% Plot true vs predicted series for train and test data, ordered in time.
    ev = evalfisOptions('OutOfRangeInputValueMessage', 'none', ...
                        'NoRuleFiredMessage', 'none');
    Sets = {'train', 'test'};
    figure('Name', char(Label));
    for s = 1:2
        X   = Data.(['X_' Sets{s}]);
        Y   = Data.(['Y_' Sets{s}]);
        idx = Data.([Sets{s} 'Idx']);

        t = idx(:) + max(delays);          % row number -> time index of x
        [t, order] = sort(t);              % random split -> re-order in time
        Yhat = evalfis(FIS, X(order, :), ev);

        subplot(2, 1, s)
        plot(t, Y(order), 'b-', 'LineWidth', 1, 'DisplayName', 'True'); hold on
        plot(t, Yhat, 'r.', 'MarkerSize', 8, 'DisplayName', 'FIS prediction'); hold off
        xlabel('Time index t'), ylabel('x(t)'), legend('Location', 'best'), grid on
        title(sprintf('%s data - %s', Sets{s}, Label), 'Interpreter', 'none')
    end
end