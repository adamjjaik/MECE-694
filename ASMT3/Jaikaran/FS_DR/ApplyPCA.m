function Z = ApplyPCA(PCAModel, X)
% ApplyPCA  Project data onto the kept principal components of a fitted PCA.
%   Z = ApplyPCA(PCAModel, X) returns the n x m PCA scores of X.
Z = (X - PCAModel.Mu) * PCAModel.Coeff;
end
