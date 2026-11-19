%% HAR_Pipeline.m: Feature Selection & Dimension Reduction Pipeline (UCI HAR)
% Run top to bottom. All settings live in the CONFIG block below.
clc; clear; close all

%% CONFIG
ScriptDir             = fileparts(mfilename('fullpath'));          % folder containing this script
Config.DataDir        = fullfile(ScriptDir, 'UCI HAR Dataset');    % path to the extracted dataset
Config.OutDir         = fullfile(ScriptDir, 'results');            % figures/tables are saved here
Config.Seed           = 1;                                         % random seed (reported in the README)
Config.NSub           = 3000;                                      % balanced subsample size (500 per activity)
Config.NumActivities  = 6;                                         % number of activity classes
Config.KPCA(1).Type = 'Gaussian';   Config.KPCA(1).Mult = 0.5;   % sigma = Mult x median distance
Config.KPCA(2).Type = 'Laplacian';  Config.KPCA(2).Mult = 1.0;
Config.RunTSNE      = true;          % optional nonlinear comparison (about 1 min)
Config.kNP          = 10;            % neighbours used for the preservation metric
rng(Config.Seed);
if ~exist(Config.OutDir, 'dir'), mkdir(Config.OutDir); end

%% STAGE 1: Data ingestion & sanity checks
tic
HAR = LoadHAR(Config.DataDir);
fprintf('Loaded in %.1f s\n', toc);
[n, p] = size(HAR.X);

% Basic properties
fprintf('\n--- Dataset properties ---\n');
fprintf('Samples: %d   Features: %d\n', n, p);
fprintf('All features numeric (double): %d\n', isa(HAR.X, 'double'));
fprintf('Missing values (NaN): %d\n', nnz(isnan(HAR.X)));
fprintf('Value range: [%.3f, %.3f]\n', min(HAR.X(:)), max(HAR.X(:)));
fprintf('Constant features (zero std): %d\n', nnz(std(HAR.X) == 0));
fprintf('Duplicate feature names: %d\n', numel(HAR.FeatureNames) - numel(unique(HAR.FeatureNames)));

% Outliers: fraction of entries more than 3 scaled MADs from the feature median
OutlierFrac = mean(isoutlier(HAR.X), 'all');
fprintf('Entries flagged as outliers (3 scaled MAD): %.2f%%\n', 100*OutlierFrac);

% Label distribution (labels are used only for colouring plots)
Counts = accumarray(HAR.y, 1);
LabelTable = table(HAR.ActivityNames, Counts, 100*Counts/n, ...
    'VariableNames', {'Activity', 'Count', 'Percent'});
disp(LabelTable)
writetable(LabelTable, fullfile(Config.OutDir, 'LabelDistribution.csv'));

%% STAGE 2: Preprocessing
% No missing values were found, so no imputation is required.
% Standardize each feature to zero mean and unit variance. Distance- and
% variance-based methods (PCA, KPCA, Laplacian Score, t-SNE) otherwise give
% more weight to features with larger spread, regardless of relevance.
FeatStd = std(HAR.X);
fprintf('\nFeature std before standardizing: min %.4f, max %.4f (ratio %.0f)\n', ...
        min(FeatStd), max(FeatStd), max(FeatStd)/min(FeatStd));

[Xz, Prep.Mu, Prep.Sigma] = zscore(HAR.X);     % Xz = (X - Mu) ./ Sigma
fprintf('After standardizing: max |mean| = %.2e, std range = [%.3f, %.3f]\n', ...
        max(abs(mean(Xz))), min(std(Xz)), max(std(Xz)));

% Save the transformation so it can be reused on new data (best practice)
save(fullfile(Config.OutDir, 'Preprocessing.mat'), 'Prep');

%% Balanced subsample (shared by all methods for plots and metrics)
NPerClass = Config.NSub / Config.NumActivities;
SubIdx = [];
for c = 1:Config.NumActivities
    idx = find(HAR.y == c);
    SubIdx = [SubIdx; idx(randperm(numel(idx), NPerClass))];   %#ok<AGROW>
end
XzSub = Xz(SubIdx, :);
ySub  = HAR.y(SubIdx);
fprintf('\nSubsample: %d points (%d per activity)\n', numel(SubIdx), NPerClass);

Emb     = struct();   % embeddings from each method (rows = subsample points)
Runtime = struct();   % run time of each method (s)

%% STAGE 3: Feature selection - Laplacian Score (unsupervised filter)
% Scores each feature by how well it preserves the local neighbourhood
% structure of the data; no labels are used. Larger score = more important.
% The similarity between neighbours is exp(-(distance/KernelScale)^2). The
% default KernelScale = 1 makes every similarity ~0 in 561 dimensions (all
% scores equal), so the scale is set to the typical neighbour distance.
tic
kNN = ceil(log(size(XzSub, 1)));                       % fsulaplacian's default neighbour count
[~, Dk] = knnsearch(XzSub, XzSub, 'K', kNN + 1);       % +1: each point is its own nearest neighbour
LS_Scale = median(Dk(:, end));                         % median distance to the kNN-th neighbour
[IdxLS, ScoreLS] = fsulaplacian(XzSub, 'NumNeighbors', kNN, 'KernelScale', LS_Scale);
Runtime.LaplacianScore = toc;

fprintf('\nLaplacian Score (%.2f s): kernel scale = %.2f, %d neighbours\n', ...
        Runtime.LaplacianScore, LS_Scale, kNN);
fprintf('Score range: [%.4f, %.4f], %d distinct values\n', ...
        min(ScoreLS), max(ScoreLS), numel(unique(round(ScoreLS, 6))));

% Top-ranked features (column index shown, since some names are duplicated)
TopN = 10;
fprintf('Top %d features:\n', TopN);
for k = 1:TopN
    j = IdxLS(k);
    fprintf('  %2d. [%3d] %-35s score %.4f\n', k, j, HAR.FeatureNames{j}, ScoreLS(j));
end

% Sorted scores of all features
figure(1); clf
bar(ScoreLS(IdxLS)); grid on
xlabel('Feature rank'); ylabel('Laplacian Score');
title('Laplacian Score of all 561 features (sorted)');
exportgraphics(gcf, fullfile(Config.OutDir, 'LaplacianScores.png'));

Emb.LS = XzSub(:, IdxLS(1:3));        % top 3 original features (use 1:2 for 2D plots)

%% STAGE 4a: Dimension reduction - PCA (linear)
tic
[CoeffPCA, ScorePCA, LatentPCA] = pca(Xz);       % fit on the FULL standardized dataset
ExplPCA = 100 * LatentPCA / sum(LatentPCA);      % % variance explained per component
Runtime.PCA = toc;
CumExpl = cumsum(ExplPCA);
fprintf('\nPCA (%.2f s): PC1-3 explain %.1f%%, %.1f%%, %.1f%% (cumulative %.1f%%)\n', ...
        Runtime.PCA, ExplPCA(1:3), CumExpl(3));
fprintf('Components needed for 90%% / 95%% variance: %d / %d\n', ...
        find(CumExpl >= 90, 1), find(CumExpl >= 95, 1));

figure(2); clf
plot(1:100, CumExpl(1:100), '-', 'LineWidth', 1.5); grid on
yline(90, '--', '90%'); yline(95, '--', '95%');
xlabel('Number of principal components'); ylabel('Cumulative explained variance (%)');
title('PCA cumulative explained variance (standardized HAR data)');
saveas(gcf, fullfile(Config.OutDir, 'PCA_ExplainedVariance.png'));

Emb.PCA = ScorePCA(SubIdx, 1:3);      % same subsample points, first 3 components

%% STAGE 4b (tuning): Kernel PCA parameter sweep
% Kernel parameters are chosen by neighbourhood preservation (k = 10) of the
% 3D embedding - an unsupervised criterion, so labels are not used.
kNP  = 10;
dEmb = 3;
Dmed = median(pdist(XzSub));          % typical distance between subsample points
fprintf('\nMedian pairwise distance (subsample) = %.2f\n', Dmed);

% Reference values: PCA and linear KPCA (should match)
NP_PCA = NeighborhoodPreservation(XzSub, Emb.PCA, kNP);
NP_Lin = NeighborhoodPreservation(XzSub, RunKPCA(XzSub, 'Linear', [], dEmb), kNP);
fprintf('Neighbourhood preservation: PCA %.3f, linear KPCA %.3f\n', NP_PCA, NP_Lin);

Sweep = { 'Gaussian',   Dmed*[0.25 0.5 1 2 4];
          'Laplacian',  Dmed*[0.25 0.5 1 2 4];
          'Polynomial', [2 3 4] };
SweepResults = [];
for k = 1:size(Sweep, 1)
    for prm = Sweep{k, 2}
        tic
        Z  = RunKPCA(XzSub, Sweep{k,1}, prm, dEmb);
        t  = toc;
        NP = NeighborhoodPreservation(XzSub, Z, kNP);
        fprintf('%-10s param = %8.3f  ->  NP = %.3f  (%.2f s)\n', Sweep{k,1}, prm, NP, t);
        SweepResults = [SweepResults; {Sweep{k,1}, prm, NP, t}];   %#ok<AGROW>
    end
end
SweepTable = cell2table(SweepResults, 'VariableNames', {'Kernel','Parameter','NeighPres','Time_s'});
writetable(SweepTable, fullfile(Config.OutDir, 'KPCA_ParameterSweep.csv'));

%% STAGE 4b (visual check): KPCA embeddings at different parameters
Check = { 'Gaussian',   0.5*Dmed;  'Gaussian',   1*Dmed;  'Gaussian',   4*Dmed;
          'Laplacian',  1*Dmed;    'Polynomial', 2;       'Polynomial', 3 };
figure(3); clf
tiledlayout(2, 3, 'TileSpacing', 'compact');
for k = 1:size(Check, 1)
    nexttile
    Z = RunKPCA(XzSub, Check{k,1}, Check{k,2}, 2);
    if strcmp(Check{k,1}, 'Polynomial')
        lbl = sprintf('%s, degree %d', Check{k,1}, Check{k,2});
    else
        lbl = sprintf('%s, \\sigma = %.2f x median', Check{k,1}, Check{k,2}/Dmed);
    end
    PlotEmbedding2D(Z, ySub, HAR.ActivityNames, lbl, k == size(Check, 1));
end
exportgraphics(gcf, fullfile(Config.OutDir, 'KPCA_ParameterCheck.png'));

%% STAGE 4b (final): Kernel PCA with the chosen kernels
for k = 1:numel(Config.KPCA)
    prm = Config.KPCA(k).Mult * Dmed;
    tic
    Z = RunKPCA(XzSub, Config.KPCA(k).Type, prm, 3);
    Runtime.(['KPCA_' Config.KPCA(k).Type]) = toc;
    Emb.(['KPCA_' Config.KPCA(k).Type]) = Z;
end

% PCA timed on the same subsample, for a fair runtime comparison
tic; [~, ~] = pca(XzSub); Runtime.PCA_subsample = toc;

%% STAGE 4c (optional): t-SNE, a nonlinear method that preserves local neighbourhoods
if Config.RunTSNE
    tic
    Emb.tSNE = tsne(XzSub, 'NumDimensions', 3, 'Perplexity', 30);
    Runtime.tSNE = toc;
end

%% STAGE 5: Visualization and quantitative comparison
MethodKeys   = {'LS', 'PCA', ['KPCA_' Config.KPCA(1).Type], ['KPCA_' Config.KPCA(2).Type]};
MethodLabels = {'Laplacian Score (top features)', 'PCA', ...
                sprintf('KPCA %s (\\sigma = %.1f x median)', Config.KPCA(1).Type, Config.KPCA(1).Mult), ...
                sprintf('KPCA %s (\\sigma = %.1f x median)', Config.KPCA(2).Type, Config.KPCA(2).Mult)};
if Config.RunTSNE
    MethodKeys{end+1} = 'tSNE';  MethodLabels{end+1} = 't-SNE (perplexity 30)';
end
nMeth = numel(MethodKeys);

% 2D embeddings (first two dimensions of each method)
figure(4); clf
tiledlayout(2, 3, 'TileSpacing', 'compact');
for m = 1:nMeth
    nexttile
    PlotEmbedding2D(Emb.(MethodKeys{m}), ySub, HAR.ActivityNames, MethodLabels{m}, m == nMeth);
    if strcmp(MethodKeys{m}, 'LS')
        xlabel(sprintf('[%d] %s', IdxLS(1), HAR.FeatureNames{IdxLS(1)}), 'Interpreter', 'none');
        ylabel(sprintf('[%d] %s', IdxLS(2), HAR.FeatureNames{IdxLS(2)}), 'Interpreter', 'none');
    elseif strcmp(MethodKeys{m}, 'PCA')
        xlabel(sprintf('PC1 (%.1f%%)', ExplPCA(1))); ylabel(sprintf('PC2 (%.1f%%)', ExplPCA(2)));
    end
end
exportgraphics(gcf, fullfile(Config.OutDir, 'Embeddings_2D.png'));

% 3D embeddings: PCA (linear) vs. the first chosen KPCA kernel (nonlinear)
figure(5); clf
tiledlayout(1, 2, 'TileSpacing', 'compact');
nexttile; PlotEmbedding3D(Emb.PCA, ySub, HAR.ActivityNames, 'PCA (3D)', false);
nexttile; PlotEmbedding3D(Emb.(MethodKeys{3}), ySub, HAR.ActivityNames, [MethodLabels{3} ' (3D)'], true);
exportgraphics(gcf, fullfile(Config.OutDir, 'Embeddings_3D.png'));

% Quantitative comparison (3D embeddings, same 3,000 points)
NP = zeros(nMeth, 1);  RT = zeros(nMeth, 1);
for m = 1:nMeth
    NP(m) = NeighborhoodPreservation(XzSub, Emb.(MethodKeys{m}), Config.kNP);
    key = MethodKeys{m};
    if strcmp(key, 'LS'),  key = 'LaplacianScore'; end
    if strcmp(key, 'PCA'), key = 'PCA_subsample';  end
    RT(m) = Runtime.(key);
end
Results = table(MethodLabels', NP, RT, 'VariableNames', {'Method', 'NeighPres_k10', 'Runtime_s'});
fprintf('\nRandom-embedding baseline for neighbourhood preservation: %.4f\n', Config.kNP/numel(ySub));
disp(Results)
writetable(Results, fullfile(Config.OutDir, 'MethodComparison.csv'));