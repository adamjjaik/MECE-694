%% (1) Initialization
clear
clc

% Generate the observations (x, y)
theta_0t = 0; 
theta_1t = 1;
theta_2t = 1.1;
[X1,X2] = meshgrid(0:0.1:10,0:0.1:10);
Y = theta_0t + theta_1t .* X1 .* sin(4*X1) ...
    + theta_2t .* X2 .* sin(2*X2)  ...
    + (-0.5 + 1 * rand(size(X1))); % adding noise to data

x = [X1(:) X2(:)];
y = Y(:);

% Create augmented x
x_au = [ones(size(x,1),1) x];

figure(1)
surf(X1, X2, Y)
xlabel('x1'), ylabel('x2'), zlabel('y')
axis([0 10 0 10 -20 20])

%% (2) Objective Function Calculation
% Obj. value for true theta
theta = [0 1 1.1]';
J = Nonlinear_ComputeCostwGrad(x_au, y, theta);
disp(['Obj. Value for True Param. = ' num2str(J)])

% Obj. value for some random theta
theta = [4.5 1.1 2.1]';
J = Nonlinear_ComputeCostwGrad(x_au, y, theta);
disp(['Obj. Value for Radom Param. = ' num2str(J)])

%% (3) Gradient Descent
clc
% Initialize fitting parameters
theta0 = zeros(3, 1); % [0 0 0]

% Gradient descent settings
num_iter = 1000;
alpha = 0.05;

disp('Running Gradient Descent ...')
% run gradient descent
[theta, J] = Nonlinear_GradientDescent2(x_au, y, theta0, alpha, num_iter);

% return results
disp(['Theta found by gradient descent: ' ...
    'Theta0 = ' num2str(theta(1)) ', Theta1 = ' num2str(theta(2))]);
disp(['Expected Theta values: ' ...
    'Theta0 = ' num2str(theta_0t) ', Theta1 = ' num2str(theta_1t)]);

figure(2)
plot(1:num_iter, J)
xlabel('# of iterations')
ylabel('Obj. function value')

%% (4) Advanced Optimization: fminunc
clc
options = optimoptions('fminunc', ...
    'Algorithm','trust-region', ...
    'SpecifyObjectiveGradient',true, ...
    'Display','iter');

theta0 = zeros(3, 1); % [0 0 0]
fun = @(theta) Nonlinear_ComputeCostwGrad(x_au, y, theta);
theta = fminunc(fun, theta0, options);


