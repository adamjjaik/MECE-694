%% HAR_Classification.m: From FS/DR to Classification & Clustering (UCI HAR)
% MECE 694 - Assignment 3, Part 4
% Extends the Assignment 2 pipeline (LoadHAR.m) to downstream
% classification and clustering, using PCA as the only FS/DR step.
% Run top to bottom. All settings live in the CONFIG block below.
clc; clear; close all

%% CONFIG
ScriptDir           = fileparts(mfilename('fullpath'));          % folder containing this script
Config.DataDir      = fullfile(ScriptDir, 'UCI HAR Dataset');    % path to the extracted dataset
Config.OutDir       = fullfile(ScriptDir, 'results_A3');         % figures/tables are saved here
Config.Seed         = 1;                                         % random seed (fixed for reproducibility)
Config.SplitFrac    = [0.6 0.2 0.2];                             % train / validation / test fractions
Config.VarThreshold = 95;                                        % PCA: cumulative explained variance (%)
Config.NumActivities = 6;                                        % number of activity classes
Config.KFolds       = 3;                                         % folds for cross-validation on train
Config.L2Grid       = [1e-4 1e-3 1e-2];                          % L2 values tried on the best MLP
rng(Config.Seed);
if ~exist(Config.OutDir, 'dir'), mkdir(Config.OutDir); end

%% STAGE 4.1a: Load data (loader from Assignment 2)
HAR = LoadHAR(Config.DataDir);
[n, p] = size(HAR.X);
fprintf('Loaded HAR: %d samples, %d features, %d activities\n', n, p, numel(HAR.ActivityNames));

%% STAGE 4.1b: Stratified 60/20/20 train/validation/test split
Split = StratifiedSplit(HAR.y, Config.SplitFrac, Config.Seed);

Xtr = HAR.X(Split.Train, :);  ytr = HAR.y(Split.Train);
Xva = HAR.X(Split.Val,   :);  yva = HAR.y(Split.Val);
Xte = HAR.X(Split.Test,  :);  yte = HAR.y(Split.Test);

% Check that every split has the same class proportions (stratification)
Counts = [accumarray(ytr, 1), accumarray(yva, 1), accumarray(yte, 1)];
SplitTable = table(HAR.ActivityNames, Counts(:,1), Counts(:,2), Counts(:,3), ...
    round(100*Counts(:,1)/numel(ytr), 1), round(100*Counts(:,2)/numel(yva), 1), ...
    round(100*Counts(:,3)/numel(yte), 1), ...
    'VariableNames', {'Activity', 'Train', 'Val', 'Test', 'Train_pct', 'Val_pct', 'Test_pct'});
fprintf('\nSplit sizes: train %d, val %d, test %d\n', numel(ytr), numel(yva), numel(yte));
disp(SplitTable)
writetable(SplitTable, fullfile(Config.OutDir, 'SplitSummary.csv'));

%% STAGE 4.1c: Z-score scaling (fit on TRAIN only, apply to val/test)
% Using only training statistics prevents information from the val/test
% sets leaking into the model (data leakage).
Scaler = FitScaler(Xtr);
XtrZ = ApplyScaler(Scaler, Xtr);
XvaZ = ApplyScaler(Scaler, Xva);
XteZ = ApplyScaler(Scaler, Xte);
fprintf('Scaled: train max |mean| = %.1e (exactly 0 by construction), val max |mean| = %.3f\n', ...
    max(abs(mean(XtrZ))), max(abs(mean(XvaZ))));

%% STAGE 4.1d: PCA (fit on TRAIN only), keep components for >= 95% variance
PCAModel = FitPCA(XtrZ, Config.VarThreshold);
Ztr = ApplyPCA(PCAModel, XtrZ);
Zva = ApplyPCA(PCAModel, XvaZ);
Zte = ApplyPCA(PCAModel, XteZ);
m   = PCAModel.m;     % PCA dimensionality (used later, e.g. RBF SVM gamma = 1/m)

fprintf('\nPCA: %d of %d components kept, cumulative explained variance = %.2f%%\n', ...
    m, p, PCAModel.CumVar(m));

% Cumulative explained variance plot (report item 1)
figure(1); clf
plot(PCAModel.CumVar, 'LineWidth', 1.5); hold on
yline(Config.VarThreshold, '--', sprintf('%d%%', Config.VarThreshold));
xline(m, ':', sprintf('m = %d', m), 'LabelVerticalAlignment', 'bottom');
plot(m, PCAModel.CumVar(m), 'ro', 'MarkerFaceColor', 'r');
hold off; grid on
xlim([1 200])
xlabel('Number of principal components'); ylabel('Cumulative explained variance (%)');
title(sprintf('PCA on training set: %d components explain %.1f%% of variance', m, PCAModel.CumVar(m)));
exportgraphics(gcf, fullfile(Config.OutDir, 'PCA_CumulativeVariance.png'));

% Save the processed data so later stages can be rerun without repeating 4.1
save(fullfile(Config.OutDir, 'HAR_PCA_Data.mat'), 'Ztr', 'Zva', 'Zte', 'ytr', 'yva', 'yte', ...
    'm', 'Scaler', 'PCAModel', 'Split', 'Config');
ActivityNames = HAR.ActivityNames;
save(fullfile(Config.OutDir, 'HAR_PCA_Data.mat'), 'ActivityNames', '-append');

%% STAGE 4.2: Classification study
% For each classifier: compare the settings with 3-fold CV on the TRAIN set,
% lock the best one, refit it on TRAIN + VAL, and evaluate once on TEST.
Ztrva = [Ztr; Zva];
ytrva = [ytr; yva];

rng(Config.Seed);
CVP = cvpartition(ytr, 'KFold', Config.KFolds, 'Stratify', true);   % same folds for all settings

[Families, MLPFactory] = DefineClassifiers(m);
tStart = tic;
Summary = [];
for f = 1:numel(Families)
    R = RunFamily(Families(f), Ztr, ytr, Ztrva, ytrva, Zte, yte, CVP, Config);
    Summary = [Summary; R];                                         %#ok<AGROW>
end

% L2 regularization (ridge weight decay) on the best MLP setting found above
isMLP = find(contains({Summary.Name}, 'MLP'));
[~, b] = max([Summary(isMLP).CVMean]);
Best = Summary(isMLP(b));
P = Best.BestParams;
L2Family.Name = sprintf('%s + L2', Best.Name);
for L2 = Config.L2Grid
    Pl = P;  Pl.L2 = L2;
    New = struct('Label', sprintf('%s, L2 = %g', Best.BestLabel, L2), ...
        'FitPredict', MLPFactory(Pl.Hidden, Pl.Act, Pl.LR, Pl.Epochs, Pl.Batch, Pl.L2), 'Params', Pl);
    if ~isfield(L2Family, 'Settings'), L2Family.Settings = New; else, L2Family.Settings(end+1) = New; end
end
Summary = [Summary; RunFamily(L2Family, Ztr, ytr, Ztrva, ytrva, Zte, yte, CVP, Config)];
fprintf('\nClassification study finished in %.1f min\n', toc(tStart)/60);

% Results table (test set) - report item 2
nF = numel(Summary);
ResultsTable = table(string({Summary.Name})', string({Summary.BestLabel})', ...
    round(100*[Summary.CVMean]', 2), ...
    round(100*arrayfun(@(s) s.Metrics.Accuracy, Summary), 2), ...
    round(100*arrayfun(@(s) s.Metrics.Sensitivity, Summary), 2), ...
    round(100*arrayfun(@(s) s.Metrics.Specificity, Summary), 2), ...
    'VariableNames', {'Classifier', 'SelectedSetting', 'CV_Acc', 'Test_Acc', 'Test_Sens', 'Test_Spec'});
disp(ResultsTable)
writetable(ResultsTable, fullfile(Config.OutDir, 'ClassificationResults.csv'));

% Every setting tried, with its CV accuracy (for "parameter setting" notes)
CVLogTable = vertcat(Summary.CVLog);
writetable(CVLogTable, fullfile(Config.OutDir, 'CV_AllSettings.csv'));

% LaTeX rows for the report table
fprintf('\n%% ---- LaTeX rows: Classifier & Setting & Accuracy & Sensitivity & Specificity ----\n');
for i = 1:nF
    fprintf('        %s & %s & %.2f & %.2f & %.2f \\\\\n', ResultsTable.Classifier(i), ...
        ResultsTable.SelectedSetting(i), ResultsTable.Test_Acc(i), ...
        ResultsTable.Test_Sens(i), ResultsTable.Test_Spec(i));
end

% Confusion matrix of the best classifier on the test set
[~, iBest] = max(ResultsTable.Test_Acc);
ActCat = @(v) categorical(v, 1:Config.NumActivities, strrep(ActivityNames, '_', ' '));
figure(2); clf
CM = confusionchart(ActCat(yte), ActCat(Summary(iBest).yhat), ...
    'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
CM.Title = sprintf('Test set confusion matrix: %s (%s)', Summary(iBest).Name, Summary(iBest).BestLabel);
exportgraphics(gcf, fullfile(Config.OutDir, 'ConfusionMatrix_Best.png'));

save(fullfile(Config.OutDir, 'ClassificationResults.mat'), 'Summary', 'ResultsTable', 'CVLogTable');

%% STAGE 4.3: Clustering study (k-means)  (next step)


%% ======================== Local functions ============================
function R = RunFamily(Fam, Ztr, ytr, Ztrva, ytrva, Zte, yte, CVP, Config)
% Compare all settings of one classifier with K-fold CV on the training set,
% then refit the best setting on train + val and evaluate it on the test set.
fprintf('\n=== %s ===\n', Fam.Name);
nS = numel(Fam.Settings);
cvMean = zeros(nS, 1);  cvStd = zeros(nS, 1);
for s = 1:nS
    rng(Config.Seed);                      % same random state for every setting
    t0 = tic;
    [cvMean(s), cvStd(s)] = CVScore(Fam.Settings(s).FitPredict, Ztr, ytr, CVP);
    fprintf('  %-55s CV acc = %6.2f +/- %4.2f %%   (%.0f s)\n', ...
        Fam.Settings(s).Label, 100*cvMean(s), 100*cvStd(s), toc(t0));
end

% Lock the best setting, refit on train + val, evaluate once on test
[~, b] = max(cvMean);
rng(Config.Seed);
yhat = Fam.Settings(b).FitPredict(Ztrva, ytrva, Zte);
M = ClassMetrics(yte, yhat, Config.NumActivities);
fprintf('  -> Selected: %s | Test acc = %.2f %%, sens = %.2f %%, spec = %.2f %%\n', ...
    Fam.Settings(b).Label, 100*M.Accuracy, 100*M.Sensitivity, 100*M.Specificity);

R.Name       = Fam.Name;
R.BestLabel  = Fam.Settings(b).Label;
R.BestParams = Fam.Settings(b).Params;
R.CVMean     = cvMean(b);
R.CVStd      = cvStd(b);
R.yhat       = yhat(:);
R.Metrics    = M;
R.CVLog      = table(repmat(string(Fam.Name), nS, 1), string({Fam.Settings.Label})', ...
    round(100*cvMean, 2), round(100*cvStd, 2), ...
    'VariableNames', {'Classifier', 'Setting', 'CV_Acc_Mean', 'CV_Acc_Std'});
end
