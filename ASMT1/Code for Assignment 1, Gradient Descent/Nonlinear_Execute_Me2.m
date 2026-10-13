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


%% Investigate the effect of num_iter (Nonlinear f3)
clc
clear J Jtemp theta_all i
% Initialize fitting parameters
theta0 = zeros(3, 1);
% Gradient descent settings
num_iter = [1 5 20 100 500];
alpha = 0.01;
for i = 1:length(num_iter)
    [theta, Jtemp] = Nonlinear_GradientDescent2(x_au, y, theta0, alpha, num_iter(i));
    J(i) = Jtemp(end);
    theta_all(:,i) = theta;
end

% Compare against true theta = [0, 1, 1.1]
true_theta = [0; 1; 1.1];
disp('theta_all (each column = one num_iter level):')
disp(theta_all)
disp('True theta:')
disp(true_theta)
disp('Final J for each num_iter level:')
disp(J)

%% Investigate the effect of alpha (Nonlinear f3)

clear J Jtemp theta_all i
% Initialize fitting parameters
theta0 = zeros(3, 1);
% Gradient descent settings
num_iter = 100;
alpha = [1e-4 1e-3 1e-2 5e-2 1e-1];
for i = 1:length(alpha)
    [theta, Jtemp] = Nonlinear_GradientDescent2(x_au, y, theta0, alpha(i), num_iter);
    J(i) = Jtemp(end);
    theta_all(:,i) = theta;
end

disp('theta_all (each column = one alpha level):')
disp(theta_all)
disp('Final J for each alpha:')
disp(J) 

%% (4) Advanced Optimization: fminunc
clc

%% --- Trust-Region ---
options_tr = optimoptions('fminunc', ...
    'Algorithm','trust-region', ...
    'SpecifyObjectiveGradient',true, ...
    'Display','iter');
theta0 = zeros(3, 1); % [0 0 0]
fun = @(theta) Nonlinear_ComputeCostwGrad(x_au, y, theta);

tic
[theta_tr, fval_tr] = fminunc(fun, theta0, options_tr);
runtime_tr = toc;

%% --- Quasi-Newton ---
options_qn = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...
    'SpecifyObjectiveGradient',true, ...
    'Display','iter');

tic
[theta_qn, fval_qn] = fminunc(fun, theta0, options_qn);
runtime_qn = toc;

%% --- Your Implemented Gradient Descent (best parameters from Section 3) ---
best_alpha = 0.05;      % update with your actual best alpha
best_num_iter = 500;    % update with your actual best num_iter

tic
[theta_gd, J_gd] = Nonlinear_GradientDescent2(x_au, y, theta0, best_alpha, best_num_iter);
runtime_gd = toc;
fval_gd = J_gd(end);

%% --- Summary Comparison ---
disp('--- Trust-Region ---')
disp(['Theta = ' num2str(theta_tr')])
disp(['Objective Function Value = ' num2str(fval_tr)])
disp(['Runtime = ' num2str(runtime_tr) ' seconds'])

disp('--- Quasi-Newton ---')
disp(['Theta = ' num2str(theta_qn')])
disp(['Objective Function Value = ' num2str(fval_qn)])
disp(['Runtime = ' num2str(runtime_qn) ' seconds'])

disp('--- Gradient Descent (Implemented) ---')
disp(['Theta = ' num2str(theta_gd')])
disp(['Objective Function Value = ' num2str(fval_gd)])
disp(['Runtime = ' num2str(runtime_gd) ' seconds'])