function M = ClassMetrics(yTrue, yPred, K)
% ClassMetrics  Accuracy, sensitivity and specificity for a K-class problem.
%   M = ClassMetrics(yTrue, yPred, K) uses a one-vs-rest view of each class:
%     Sensitivity (recall) of class k = TP / (TP + FN)
%         -> of the samples that truly are class k, the fraction detected
%     Specificity of class k = TN / (TN + FP)
%         -> of the samples that are NOT class k, the fraction correctly rejected
%   Overall sensitivity/specificity are the macro averages over the K classes.
C  = confusionmat(yTrue(:), yPred(:), 'Order', 1:K);   % rows = true, columns = predicted
N  = sum(C(:));
TP = diag(C);
FN = sum(C, 2) - TP;
FP = sum(C, 1)' - TP;
TN = N - TP - FN - FP;

M.Confusion    = C;
M.Accuracy     = sum(TP) / N;
M.SensPerClass = TP ./ (TP + FN);
M.SpecPerClass = TN ./ (TN + FP);
M.Sensitivity  = mean(M.SensPerClass);
M.Specificity  = mean(M.SpecPerClass);
end
