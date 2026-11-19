function Split = StratifiedSplit(y, Frac, Seed)
% StratifiedSplit  Stratified train/validation/test split.
%   Split = StratifiedSplit(y, Frac, Seed) splits the samples with labels y
%   into train/validation/test sets with fractions Frac = [train val test]
%   (e.g., [0.6 0.2 0.2]). Stratified means each set keeps the same class
%   proportions as the full dataset.
%   Returns logical index vectors Split.Train, Split.Val, Split.Test.

rng(Seed);
n = numel(y);

% Step 1: hold out the test set (stratified by class)
c1 = cvpartition(y, 'HoldOut', Frac(3), 'Stratify', true);
idxRest = find(training(c1));

% Step 2: split the remainder into train and validation (stratified)
% e.g. val = 0.2 of the total = 0.25 of the remaining 0.8
c2 = cvpartition(y(idxRest), 'HoldOut', Frac(2) / (Frac(1) + Frac(2)), 'Stratify', true);

Split.Test  = test(c1);
Split.Val   = false(n, 1);  Split.Val(idxRest(test(c2)))       = true;
Split.Train = false(n, 1);  Split.Train(idxRest(training(c2))) = true;
end
