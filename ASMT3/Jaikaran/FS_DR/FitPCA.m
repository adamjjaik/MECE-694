function PCAModel = FitPCA(X, VarThreshold)
% FitPCA  Fit PCA and keep the smallest number of components whose
% cumulative explained variance is at least VarThreshold (%).
%   PCAModel = FitPCA(X, VarThreshold) fits PCA on X (rows = samples).
%   Fit on the TRAINING set only; transform any set with ApplyPCA.
%   Returns:
%     Coeff   p x m matrix of the kept principal directions
%     Mu      1 x p mean used for centring
%     m       number of components kept
%     CumVar  cumulative explained variance (%) of ALL components
% HAR features are linearly dependent (rank < p), which makes pca warn while
% computing a statistic (Hotelling T^2) that is not used here.
warning('off', 'stats:pca:ColRankDefX');
[Coeff, ~, ~, ~, Explained, Mu] = pca(X);
warning('on', 'stats:pca:ColRankDefX');
CumVar = cumsum(Explained);
m = find(CumVar >= VarThreshold, 1);

PCAModel.Coeff  = Coeff(:, 1:m);
PCAModel.Mu     = Mu;
PCAModel.m      = m;
PCAModel.CumVar = CumVar;
end
