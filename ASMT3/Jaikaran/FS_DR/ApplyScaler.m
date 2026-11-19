function Xz = ApplyScaler(Scaler, X)
% ApplyScaler  Apply a fitted z-score scaler: Xz = (X - Mu) ./ Sigma.
Xz = (X - Scaler.Mu) ./ Scaler.Sigma;
end
