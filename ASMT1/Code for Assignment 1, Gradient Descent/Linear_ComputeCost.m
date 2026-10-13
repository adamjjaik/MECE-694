function J = Linear_ComputeCost(x, y, theta)
%   J = Linear_ComputeCost(x, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in x and y

% Initialize some useful values
M = length(y); % number of observations

% You need to return the following variables correctly
J = 0;

% ************ Add You Code Here ***********
% Instructions: 
% (1) Calculate the objective function for the given x, y, and theta

% Build a matrix that consists of all 1's in the first column
% and the x values in the second
% Copmute f_hat = theta0 + theta1*x by multiplying the matricies
f_hat = x * theta;
% Copmute the difference from the actual value for each point
errors = f_hat - y;
% Objective function J using mean-squared-error
J = (1 / (2 * M)) * sum(errors.^2);

% ******************************************

end
