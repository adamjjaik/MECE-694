function Scaler = FitScaler(X)
% FitScaler  Fit a z-score scaler (mean and standard deviation per feature).
%   Scaler = FitScaler(X) computes the mean and std of each column of X
%   (rows = samples). Fit on the TRAINING set only; apply to any set with
%   ApplyScaler.
Scaler.Mu    = mean(X, 1);
Scaler.Sigma = std(X, 0, 1);
Scaler.Sigma(Scaler.Sigma == 0) = 1;    % guard: constant features are left unscaled
end
