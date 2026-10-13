function [theta, J] = Nonlinear_GradientDescent2(x, y, theta, alpha, num_iters)
% Nonlinear_GradientDescent2(X, y, theta, alpha, num_iters)
% updates theta by taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
M = length(y); % number of training examples
J = zeros(num_iters, 1);

for iter = 1:num_iters
    % ************ Add You Code Here ***********
    % Instructions: 
    % (1) implement one iteration of gradient descent and update theta
    % (2) record obje. function value J

    % Compute the gradient at the current theta (old, pre-update value).
    % discard the cost (~) since it reflects the pre-update theta.
    [~, grad] = Nonlinear_ComputeCostwGrad(x, y, theta);

    % Gradient descent update: step theta in the direction that
    % decreases the cost, scaled by the learning rate alpha
    theta = theta - alpha * grad;

    % Record the objective function value using the updated theta,
    J(iter) = Nonlinear_ComputeCostwGrad(x, y, theta);

    % ******************************************
end

end
