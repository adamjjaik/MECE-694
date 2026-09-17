%% (1) Initialization
clear
clc

% Generate the observations (x, y)
theta_0t = 0.7; 
theta_1t = 1.3;
theta_2t = 2.3;
theta_3t = 3.1;
x = [-30:0.2:30]'; %#ok
y = theta_0t + theta_1t .* abs(x) ...
    + theta_2t .* cos(x) + theta_3t .* sin(x) ...
    + (-1.5 + 3 * rand(length(x),1)); % adding noise to data

% Create augmented x
x_au = [ones(size(x,1),1) x];

figure(1)
scatter(x, y, 'b*')
ylabel('y'), xlabel('x')
axis([-20 20 -1 30])

%% (2) Objective Function Calculation
% Obj. value for true theta
theta = [0.7 1.3 2.3 3.1]';
J = Nonlinear_ComputeCost(x_au, y, theta);
disp(['Obj. Value for True Param. = ' num2str(J)])

% Obj. value for some random theta
theta = [4.5 1.1 2.1 1.8]';
J = Nonlinear_ComputeCost(x_au, y, theta);
disp(['Obj. Value for Radom Param. = ' num2str(J)])

%% (3) Gradient Descent
clc
% Initialize fitting parameters
theta0 = zeros(4, 1); % [0 0 0 0]

% Gradient descent settings
num_iter = 1000;
alpha = 0.05;

disp('Running Gradient Descent ...')
% run gradient descent
[theta, J] = Nonlinear_GradientDescent(x_au, y, theta0, alpha, num_iter);

% return results
disp(['Theta found by gradient descent: ' ...
    'Theta0 = ' num2str(theta(1)) ', Theta1 = ' num2str(theta(2))]);
disp(['Expected Theta values: ' ...
    'Theta0 = ' num2str(theta_0t) ', Theta1 = ' num2str(theta_1t)]);

% Plot the reuslts
figure(1)
hold on 
plot(x, x_au*theta, 'LineWidth', 2)
legend('Observations', 'Linear Regression')
hold off 

figure(2)
plot(1:num_iter, J)
xlabel('# of iterations')
ylabel('Obj. function value')

%% (4) Investigate the effect of random initialization
clc
clear J Jtemp i j

% Initialize fitting parameters
theta0 = [zeros(4, 1) rand(4,10) rand(4,10)+2];

% Gradient descent settings
num_iter = 10;

for i = 1:size(theta0,2) % for each theta0
    [~, Jtemp] = Nonlinear_GradientDescent(x_au, y, theta0(:,i), alpha, num_iter);
    J(i) = Jtemp(end);
end

% Plot Obj. Values for different theta0's
figure(3)
scatter(1:size(theta0,2), J)

%% (5) Investigate the effect of alpha
clc
clear J Jtemp i j

% Initialize fitting parameters
theta0 = zeros(4, 1) ;

% Gradient descent settings
num_iter = 100;

alpha = [1e-5 1e-4 1e-3 1e-2 5e-2 1e-1];
for i = 1:length(alpha)
    [~, Jtemp] = Nonlinear_GradientDescent(x_au, y, theta0, alpha(i), num_iter);
    J(i) = Jtemp(end);
end

