function [MeanAcc, StdAcc, FoldAcc] = CVScore(FitPredict, X, y, CVP)
% CVScore  K-fold cross-validation accuracy of one classifier setting.
%   [MeanAcc, StdAcc, FoldAcc] = CVScore(FitPredict, X, y, CVP)
%   FitPredict : function handle yhat = FitPredict(Xtrain, ytrain, Xtest)
%                that trains the classifier and predicts labels for Xtest
%   X, y       : data (rows = samples) and labels used for CV (training set)
%   CVP        : cvpartition object (the same folds are used for every
%                setting, so all settings are compared fairly)
K = CVP.NumTestSets;
FoldAcc = zeros(K, 1);
for k = 1:K
    tr = training(CVP, k);
    te = test(CVP, k);
    yhat = FitPredict(X(tr,:), y(tr), X(te,:));
    FoldAcc(k) = mean(yhat(:) == y(te));
end
MeanAcc = mean(FoldAcc);
StdAcc  = std(FoldAcc);
end
