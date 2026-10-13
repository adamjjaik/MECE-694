function [theta, J] = Linear_GradientDescent(x, y, theta, alpha, num_iters)
% Linear_GradientDescent(X, y, theta, alpha, num_iters)
% updates theta by taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
M = length(y); % number of training examples
J = zeros(num_iters, 1);

for iter = 1:num_iters
    % ************ Add You Code Here ***********
    % Instructions: 
    % (1) implement one iteration of gradient descent and update theta

    % Compute predictions using the current (pre-update) theta,
    % then the prediction error for each observation
    f_hat = x * theta;
    errors = f_hat - y;

    % Gradient of J w.r.t. theta
    grad = (1/M)*transpose(x)*errors;

    % Gradient descent update: step theta in the direction that
    % decreases the cost, scaled by the learning rate alpha
    theta = theta - alpha * grad;
    
    % (2) record obje. function value J
    J(iter) = Linear_ComputeCost(x, y, theta);

    % ******************************************
end

end
