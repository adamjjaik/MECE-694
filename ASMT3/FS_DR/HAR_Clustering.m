%% HAR_Clustering.m: Clustering study (k-means) on the PCA features (UCI HAR)
% MECE 694 - Assignment 3, Part 4.3
% Uses the train/val/test PCA features saved by HAR_Classification.m
% (run Stage 4.1 of that script first).
%
% k-means is fitted on the TRAINING set (no labels used). Validation/test
% samples are assigned to the nearest cluster centre. The true labels are
% used ONLY for scoring: each cluster is matched to one activity so that
% the number of correctly matched samples is maximized.
clc; clear; close all

%% CONFIG
ScriptDir        = fileparts(mfilename('fullpath'));
Config.OutDir    = fullfile(ScriptDir, 'results_A3');
Config.Seed      = 1;
Config.K         = 6;          % number of clusters (HAR ground truth: 6 activities)
Config.Replicates = 10;        % k-means restarts (best of 10 is kept)
rng(Config.Seed);

load(fullfile(Config.OutDir, 'HAR_PCA_Data.mat'), 'Ztr', 'Zte', 'ytr', 'yte', 'm', 'ActivityNames');
Names = strrep(ActivityNames, '_', ' ');
fprintf('Loaded PCA features: m = %d, %d training and %d test samples\n', m, numel(ytr), numel(yte));

%% STAGE 4.3a: k-means on the training PCA features (labels not used)
tic
[ClusterTr, Centres, SumD] = kmeans(Ztr, Config.K, 'Replicates', Config.Replicates, ...
    'MaxIter', 500, 'Display', 'off');
fprintf('k-means (k = %d, %d replicates): %.1f s, total within-cluster distance = %.3g\n', ...
    Config.K, Config.Replicates, toc, sum(SumD));

% Assign test samples to the nearest cluster centre
[~, ClusterTe] = pdist2(Centres, Zte, 'euclidean', 'Smallest', 1);
ClusterTe = ClusterTe(:);

%% STAGE 4.3b: Score with labels (scoring only)
% Match clusters to activities on the training set, then apply the same
% mapping to the test set
[Map, Contingency] = MatchClusters(ytr, ClusterTr, Config.K);
yhatTr = Map(ClusterTr);
yhatTe = Map(ClusterTe);
AccTr = mean(yhatTr == ytr);
AccTe = mean(yhatTe == yte);
fprintf('\nk-means accuracy (k = %d): train %.2f %%, test %.2f %%\n', Config.K, 100*AccTr, 100*AccTe);

% Contingency table: how each activity is spread over the clusters
% (columns ordered so that cluster j is the one matched to activity j)
[~, Order] = sort(Map);
CTable = array2table(Contingency(:, Order), 'RowNames', Names, ...
    'VariableNames', cellstr(compose('Cluster_%s', string(strrep(Names, ' ', '_')))));
fprintf('\nTraining set: activity (rows) vs matched cluster (columns)\n');
disp(CTable)
writetable(CTable, fullfile(Config.OutDir, 'KMeans_Contingency.csv'), 'WriteRowNames', true);

% Per-activity fraction correctly clustered (test set)
M = ClassMetrics(yte, yhatTe, Config.K);
PerClass = table(Names, round(100*M.SensPerClass, 1), 'VariableNames', {'Activity', 'Test_CorrectlyClustered_pct'});
disp(PerClass)

%% STAGE 4.3c: PCA-2D scatter with cluster labels (report item 3)
figure(1); clf
set(gcf, 'Position', [100 100 1200 480])
Colors = lines(Config.K);

subplot(1,2,1)
gscatter(Ztr(:,1), Ztr(:,2), categorical(yhatTr, 1:Config.K, Names), Colors, '.', 8);
hold on
plot(Centres(:,1), Centres(:,2), 'kx', 'MarkerSize', 14, 'LineWidth', 2.5, 'DisplayName', 'Cluster centres');
hold off
xlabel('PC1'); ylabel('PC2'); grid on
title(sprintf('k-means clusters (k = %d), named by matched activity', Config.K));
legend('Location', 'best', 'FontSize', 8)

subplot(1,2,2)
gscatter(Ztr(:,1), Ztr(:,2), categorical(ytr, 1:Config.K, Names), Colors, '.', 8);
xlabel('PC1'); ylabel('PC2'); grid on
title('True activity labels (for comparison)');
legend('Location', 'best', 'FontSize', 8)

sgtitle(sprintf('Training set in the first two PCs (k-means train accuracy %.1f%%)', 100*AccTr))
exportgraphics(gcf, fullfile(Config.OutDir, 'KMeans_PCA2D.png'));

% Confusion matrix of matched clusters vs true activities (test set)
figure(2); clf
ActCat = @(v) categorical(v, 1:Config.K, Names);
CM = confusionchart(ActCat(yte), ActCat(yhatTe), 'RowSummary', 'row-normalized');
CM.Title = sprintf('k-means (k = %d), test set: true activity vs matched cluster', Config.K);
exportgraphics(gcf, fullfile(Config.OutDir, 'KMeans_ConfusionMatrix.png'));

save(fullfile(Config.OutDir, 'ClusteringResults.mat'), 'ClusterTr', 'ClusterTe', 'Centres', ...
    'Map', 'Contingency', 'AccTr', 'AccTe');
