%% (1) Initialization
clear
clc

% Generate the observations (x, y)
theta_0t = 3.1; 
theta_1t = 2.5;
x = [1:0.1:10]'; %#ok
y = theta_0t + theta_1t .* x ...
    + (-3 + 6 * rand(length(x),1)); % adding noise to data

% Create augmented x
x_au = [ones(size(x,1),1) x];

figure(1)
scatter(x, y, 'b*')
ylabel('y'), xlabel('x')
axis([0 11 4 30])

%% (2) Objective Function Calculation
% Obj. value for true theta
theta = [3.1 2.5]';
J = Linear_ComputeCost(x_au, y, theta);
disp(['Obj. Value for True Param. = ' num2str(J)])

% Obj. value for some random theta
theta = [4.5 1.1]';
J = Linear_ComputeCost(x_au, y, theta);
disp(['Obj. Value for Radom Param. = ' num2str(J)])

%% (3) Gradient Descent
clc
% Initialize fitting parameters
theta0 = zeros(2, 1); % [0 0]

% Gradient descent settings
num_iter = 10000;
alpha = 0.05;

disp('Running Gradient Descent ...')
% run gradient descent
[theta, J] = Linear_GradientDescent(x_au, y, theta0, alpha, num_iter);

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
theta0 = [rand(2,10)-5, rand(2,10), rand(2,10)+3];
% Gradient descent settings
num_iter = 1000;
alpha = 0.05;
for i = 1:size(theta0,2) % for each theta0
    [~, Jtemp] = Linear_GradientDescent(x_au, y, theta0(:,i), alpha, num_iter);
    J(i) = Jtemp(end);
end
% Plot Obj. Values for different theta0's
figure
hold on
yline(1.3984, '--k')
scatter(1:10, J(1:10), 'g', 'filled')     % rand(2,10)-5 group ("far", roughly [-5,-4])
scatter(11:20, J(11:20), 'b', 'filled')   % rand(2,10) group ("medium", roughly [0,1])
scatter(21:30, J(21:30), 'r', 'filled')   % rand(2,10)+3 group ("near", roughly [3,4])
xlabel('Initial Guess Index')
ylabel('Final Objective Function Value, J')
legend('J Converged','Group 1 (far)', 'Group 2 (medium)', 'Group 3 (near)')
ytickformat('%.2f')
hold off

%% (5) Investigate the effect of alpha
clc
clear J Jtemp i j J_history
% Initialize fitting parameters
theta0 = zeros(2, 1) ;
% Gradient descent settings
num_iter = 1000000;
alpha = [1e-5 1e-4 1e-3 1e-2 5e-2 1e-1];
J_history = zeros(num_iter, length(alpha));
for i = 1:length(alpha)
    [theta, Jtemp] = Linear_GradientDescent(x_au, y, theta0, alpha(i), num_iter);
    J(i) = Jtemp(end);
    theta_all(:,i) = theta;
    J_history(:,i) = Jtemp;
end

% Identify which alphas diverged (Inf/NaN anywhere in their J history)
diverged = any(~isfinite(J_history), 1);

% Plot objective function value vs iterations, excluding diverged alphas
figure(5)
plot(1:num_iter, J_history(:, ~diverged), 'LineWidth', 1.5)
xlabel('Number of Iterations')
ylabel('Objective Function Value, J')
legend(arrayfun(@(a) ['\alpha = ' num2str(a)], alpha(~diverged), 'UniformOutput', false))
title('Convergence of J for Different Learning Rates')

if any(diverged)
    disp('The following alpha values diverged (excluded from plot):')
    disp(alpha(diverged))
end


%% (6) Investigate the effect of iterations
clc
clear J Jtemp i j

% Initialize fitting parameters
theta0 = zeros(2, 1) ;

% Gradient descent settings
num_iter = [10 100 1000 10000 100000];

alpha = 0.01;
for i = 1:length(num_iter)
    [theta, Jtemp] = Linear_GradientDescent(x_au, y, theta0, alpha, num_iter(i));
    J(i) = Jtemp(end);
    theta_all(:,i) = theta;
end

disp(theta_all)

