function [y_hat] = PredictMLPOutputs(theta1, theta2, x)
% This function returns the label associated with each observation using
% the trained two-layer MLP with optimal wieghts theta1 and theta2 

% Useful values
M = size(x, 2);
NumClasses = size(theta2, 1);

% Output of Hidden Layer
f1 = sigmf(theta1 * [ones(1, M); x], [1 1]);

% Calculate Output of Output Layer
f2 = sigmf(theta2 * [ones(1, M); f1], [1 1]);

% Identify the class label
[~, y_hat] = max(f2, [], 1);

y_hat = [y_hat; y_hat]; 
y_hat(1, y_hat(1,:) == 2) = 0;
y_hat(2, y_hat(2,:) == 1) = 0;
y_hat(2, y_hat(2,:) == 2) = 1;
end
