%% (1) Initialization and Problem Creation
clc
clear

% BodyFat or BreastCancer
Dataset_Name = 'BodyFat';
% Load Dataset
Data = LoadData(Dataset_Name);

%% (2) Sequential Forward/Backward Search
% Create Objectve Function
Objective = @(Input_tr, Output_tr, Input_test, Output_test) SequentialFeatureSelection(...
    Input_tr, Output_tr, Input_test, Output_test);

opts = statset('Display','iter');
[SFIndex_SFBS, history] = sequentialfs(Objective, Data.Input', Data.Output', ...
    'cv', 5 , 'options',opts);

%% (3) Laplacian Scores Method
% d --> Desired Number of Features 
d = 1; 

[Index_LS, LSs] = fsulaplacian(Data.Input','Distance', 'seuclidean');

% Plot Results
figure(1)
bar(LSs(Index_LS))
xlabel('Feature Rank')
ylabel('Feature Importance Score')  
SFIndex_LS = zeros(1,length(Index_LS));
SFIndex_LS(Index_LS(1:d)) = 1;

%% (4) Best Individual d Features (Complete This Section)
% Weight for Error in Training Data (from 0 to 1)
WeightTrain = 0.5;
% Number of Artificial Neural Network Evaluations
NumExecutions = 20;
% Create Objectve Function
Objective = @(Index) FeatureSelectionObjective(...
    Index, Data, WeightTrain, NumExecutions);

% d --> Desired Number of Features 
d = 3; 

[SFIndex_BIdF, Error] = BestInd_d_Features_Incomplete(d, Objective, Data);

%% (5) Principal Component Analysis
% d --> Desired Number of Features 
d = 3;

% Apply PCA
[~, score, ~, ~,explained, ~] = pca(Data.Input'); 
plot(1:Data.NumFeatures, cumsum(explained), '-x')
xticks(1:Data.NumFeatures), grid on

% Create Transformed Input
Data_PCA = Data;
Data_PCA.Input = score(:,1:d)';
Data_PCA.d = d;

%% (6) Kernel Principal Component Analysis (Complete This Section)
% d --> Desired Number of Features 
d = 3;

% Kernel Type: Linear, Polynomial, Gaussian, etc.
Kernel.Type = 'Linear';
Kernel.Parameter = 1;

% Apply Kernel PCA
Data_KPCA = Data;
Data_KPCA.Input = KernelPCA_Incomplete(Data.Input', Kernel, d); 
Data_KPCA.d = d;

%% (7) Comparison of Methods (self-contained; only Section 1 must be run first)
% ---- 7a. Setup ----
rng(1);                          % reproducible ANN initialisation and splits
WeightTrain   = 0.5;
NumExecSelect = 20;              % ANN runs inside Best Individual d Features
NumExecEval   = 50;              % ANN runs for the final reported error
                                 % (use 2 for both while debugging)
dList = 1:5;
nD    = numel(dList);

% Kernels for the three KPCA rows
% NOTE: kernels 2 and 3 are PLACEHOLDERS; replace after your tuning pass
Kernels(1).Name = 'KPCA Linear';      Kernels(1).Type = 'Linear';     Kernels(1).Parameter = 1;
Kernels(2).Name = 'KPCA Polynomial';  Kernels(2).Type = 'Polynomial'; Kernels(2).Parameter = 2;
Kernels(3).Name = 'KPCA Gaussian';    Kernels(3).Type = 'Gaussian';   Kernels(3).Parameter = 100;

% Row order: 1 = LS, 2 = BIdF, 3 = PCA, 4..6 = KPCA kernels
MethodNames = {'Laplacian Scores', 'Best Individual d', 'PCA', Kernels.Name};
nM = numel(MethodNames);

Err          = zeros(nM, nD);    % error table
Time         = zeros(nM, nD);    % run time of the selection/reduction step
Proj2D       = cell(nM, 1);      % d = 2 data, saved for the scatter plots
SelectedFeat = cell(2, nD);      % which original features LS / BIdF chose

% Helpers: error of a dataset/selection on the final evaluation, and the
% cheaper objective used inside Best Individual d Features
EvalErr         = @(Dt, idx) FeatureSelectionObjective(idx, Dt, WeightTrain, NumExecEval);
ObjectiveSelect = @(idx) FeatureSelectionObjective(idx, Data, WeightTrain, NumExecSelect);

%% ---- 7b. Baseline (all features) ----
tic
Err_Baseline  = EvalErr(Data, ones(1, Data.NumFeatures));
Time_Baseline = toc;             % = cost of one full ANN evaluation, for reference

%% ---- 7c. Sequential Forward/Backward Search (cv = 5 and 10) ----
cvList = [5 10];
for k = 1:numel(cvList)
    tic
    SFIdx = sequentialfs(@SequentialFeatureSelection, Data.Input', Data.Output', ...
        'cv', cvList(k));
    SFBS(k).Time        = toc;
    SFBS(k).cv          = cvList(k);
    SFBS(k).NumSelected = sum(SFIdx);
    SFBS(k).Selected    = find(SFIdx);
    SFBS(k).Err         = EvalErr(Data, SFIdx);
end

%% ---- 7d. Main loop over d ----
for di = 1:nD
    d = dList(di);

    % --- Laplacian Scores (row 1)
    tic
    [Index_LS, ~] = fsulaplacian(Data.Input', 'Distance', 'seuclidean');
    SF_LS = zeros(1, Data.NumFeatures);
    SF_LS(Index_LS(1:d)) = 1;
    Time(1, di) = toc;
    Err(1, di)  = EvalErr(Data, SF_LS);
    SelectedFeat{1, di} = find(SF_LS);
    if d == 2, Proj2D{1} = Data.Input(logical(SF_LS), :); end

    % --- Best Individual d Features (row 2)
    tic
    [SF_BIdF, ~] = BestInd_d_Features_Incomplete(d, ObjectiveSelect, Data);
    Time(2, di) = toc;
    Err(2, di)  = EvalErr(Data, SF_BIdF);
    SelectedFeat{2, di} = find(SF_BIdF);
    if d == 2, Proj2D{2} = Data.Input(logical(SF_BIdF), :); end

    % --- PCA (row 3)
    tic
    [~, score] = pca(Data.Input');
    Data_PCA = Data;
    Data_PCA.Input = score(:, 1:d)';           % first d components, features x observations
    Data_PCA.d = d;
    Time(3, di) = toc;
    Err(3, di)  = EvalErr(Data_PCA, ones(1, d));
    if d == 2, Proj2D{3} = Data_PCA.Input; end

    % --- Kernel PCA (rows 4..6)
    for k = 1:numel(Kernels)
        m = 3 + k;
        tic
        Data_K = Data;
        Data_K.Input = KernelPCA_Incomplete(Data.Input', Kernels(k), d);
        Data_K.d = d;
        Time(m, di) = toc;
        Err(m, di)  = EvalErr(Data_K, ones(1, d));
        if d == 2, Proj2D{m} = Data_K.Input; end
    end

    fprintf('Finished d = %d\n', d);           % progress indicator
end

%% ---- 7e. Results ----
varNames  = compose('d%d', dList);
ErrTable  = array2table(Err,  'RowNames', MethodNames, 'VariableNames', varNames);
TimeTable = array2table(Time, 'RowNames', MethodNames, 'VariableNames', varNames);
disp('Error (weighted train + test MSE):'); disp(ErrTable)
disp('Run time of selection/reduction step (s):'); disp(TimeTable)
fprintf('Baseline error = %.4f (%d features), ANN evaluation time = %.1f s\n', ...
    Err_Baseline, Data.NumFeatures, Time_Baseline);
for k = 1:numel(SFBS)
    fprintf('SFBS cv=%d: %d features selected, error = %.4f, time = %.1f s\n', ...
        SFBS(k).cv, SFBS(k).NumSelected, SFBS(k).Err, SFBS(k).Time);
end

save('Q3_Results.mat', 'Err', 'Time', 'Proj2D', 'SelectedFeat', 'SFBS', ...
     'Err_Baseline', 'Time_Baseline', 'MethodNames', 'Kernels');

%% ---- 7f. d = 2 scatter plots: best and worst method ----
col2 = find(dList == 2);                       % column of the d = 2 results
[~, bestM]  = min(Err(:, col2));
[~, worstM] = max(Err(:, col2));

figure(10); clf
plotIdx = [bestM, worstM];
plotTag = {'Best', 'Worst'};
for p = 1:2
    m = plotIdx(p);
    subplot(1, 2, p)
    scatter(Proj2D{m}(1,:), Proj2D{m}(2,:), 25, Data.Output, 'filled');
    colorbar; grid on
    if m <= 2    % feature selection: axes are original features
        xlabel(sprintf('Feature %d', SelectedFeat{m, col2}(1)));
        ylabel(sprintf('Feature %d', SelectedFeat{m, col2}(2)));
    else         % dimension reduction: axes are components
        xlabel('Component 1'); ylabel('Component 2');
    end
    title(sprintf('%s: %s (error = %.4f)', plotTag{p}, MethodNames{m}, Err(m, col2)));
end