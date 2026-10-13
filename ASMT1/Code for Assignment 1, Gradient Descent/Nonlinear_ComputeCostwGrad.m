function [J, gradJ] = Nonlinear_ComputeCostwGrad(x, y, theta)
%  J = Linear_ComputeCost(x, y, theta) computes the
% (1) objective function using theta as the parameter and
% (2) value of gradient uinsg theta and x


% Initialize some useful values
M = length(y); % number of observations

% You need to return the following variables correctly
J = 0;
gradJ = zeros(length(theta),1);

% ************ Add You Code Here ***********
% Instructions:
% (1) Calculate the objective function for the given x, y, and theta
% (2) Calculate the gradient value for the given x, y, and theta

% Extract raw features x1 and x2 from the augmented input matrix
% (column 1 of x is the ones column, columns 2 and 3 are x1 and x2)
x1 = x(:,2);
x2 = x(:,3);

% Build the nonlinear feature transformations so that theta enters
% the model linearly: h = theta0 + theta1*z1 + theta2*z2
z1 = x1 .* sin(4*x1);
z2 = x2 .* sin(2*x2);

% Assemble the design matrix Z (M x 3): ones column + z1 + z2
Z = [ones(M,1), z1, z2];

% Compute predictions and errors, same pattern as linear regression
f_hat = Z * theta;
errors = f_hat - y;
J = (1/(2*M)) * sum(errors.^2);

% Objective function: mean-squared error (always computed,
% regardless of how many outputs are requested)

if nargout > 1 % if gradient required, then compute
    gradJ = (1/M) * Z' * errors;
end

% ******************************************

end
