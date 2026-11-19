function [Inputs, Targets] = TimeSeries_Data(x, delays)
% TimeSeries_Data  Build input/output pairs for time-series prediction
%   x      : time series (vector)
%   delays : vector of delays, e.g. [6 12 18 24]
%   Inputs : (T - max(delays)) x numel(delays) matrix
%   Targets: (T - max(delays)) x 1 vector

    x = x(:);                 % force column vector
    T = numel(x);             % length of the series
    maxD = max(delays);       % largest delay sets the first usable t

    t = (maxD + 1 : T)';      % all valid time indices

    nD = numel(delays);
    Inputs = zeros(numel(t), nD);
    for k = 1:nD
        Inputs(:, k) = x(t - delays(k));   % column k = signal delayed by delays(k)
    end

    Targets = x(t);           % output = current value at each t
end