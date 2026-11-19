%% (1) Create Data
clear
clc
close all

% x: input data --> p by M (M: # of observations, p: # of features)
% Class 1
Mu1 = [5 5]; Sigma1 = [3  0.1; 0.1  3];
X1 = mvnrnd(Mu1, Sigma1, 50)';

Mu1 = [6 -2]; Sigma1 = [2  0; 0  2];
X1 = [X1 mvnrnd(Mu1, Sigma1, 50)'];

% Class 2
Mu2 = [1 1]; Sigma2 = [5  0.5; 0.5  5];
X2 = mvnrnd(Mu2, Sigma2, 100)';

x = [X1'; X2']';
t_org = [ones(size(X1,2),1); -1*ones(size(X2,2),1)]';

figure(1)
plot(X1(1,:), X1(2,:), 'b*', 'DisplayName','Class 1');
hold on
plot(X2(1,:), X2(2,:), 'r+', 'DisplayName','Class 2');
hold off
ylabel('x_2'), xlabel('x_1')
axis([-4 12 -4 11])
legend

clearvars -except x t_org

%% (2) Kernel Support Vectir Machine (KSVM), MATLAB
clc
% Define SVM Parameter
C = 2;

%%% Design SVM
SVM = fitcsvm(x', t_org', ...
    'boxconstraint', C, ...
    'KernelFunction', 'gaussian', ... % Options are: gaussian, linear, polynomial
    'Crossval', 'off', ... % Options are: off, on
    'Solver', 'ISDA'); % Options are: ISDA, L1QP, SMO
    
%%% Test SVM
y = predict(SVM, x')';

% Creating proper output format for Confusion Matrix
t = [t_org; t_org];
t(1,t(1,:) == -1) = 0;
t(2,t(2,:) == 1) = 0; 
t(2,t(2,:) == -1) = 1;

y = [y; y];
y(1,y(1,:) == -1) = 0;
y(2,y(2,:) == 1) = 0; 
y(2,y(2,:) == -1) = 1;

% Plot
figure, plotconfusion(t, y, 'All Data');
title('Performance of MATLAB KSVM Implementation')

clearvars -except x t_org

%% (3) Kernel Support Vectir Machine (KSVM), Your Code
clc
% Class labels (1 for Class 1, -1 for Class 2)
% (Section 1 only keeps "t_org", but this section uses "t")
if ~exist('t', 'var')
    t = t_org;
end

% Define SVM Parameter
C = 10;

% STEP 0: Define Kernel Function
Kernel.Type = 'Gaussian';
Kernel.Parameter = 1.5;
K = KernelFunction_Incomplete(Kernel);

%%% STEP 1: Calculate Matrix H = [h_ij]
% h_ij = y_i * y_j * Kernel(x_i, x_j)
NumObs = size(x,2);
H = zeros(NumObs, NumObs);
for i = 1:NumObs
    for j = i:NumObs
        H(i,j) = t(i) * t(j) * K(x(:,i), x(:,j));
        H(j,i) = H(i,j);      % H is symmetric, so fill the lower half too
    end
end

% Define Vector of Minus Ones
% (quadprog minimizes 0.5*a'*H*a + f'*a; f = -1 gives the "- sum(alpha)" term)
MinusOnes = -ones(NumObs, 1);

%%% STEP 2: Solve Quadratic Programming Problem
% Define Equality Constraints: sum(alpha_i * y_i) = 0
Aeq = t; beq = 0;
% Define Lower/Upper Bounds: 0 <= alpha_i <= C
lb = zeros(NumObs, 1); ub = C * ones(NumObs, 1);

% Set Optimization Parameters
options = optimset('Algorithm', 'interior-point-convex', ...
    'Display', 'iter', 'MaxIter', 100);
% Solve Quadratic Programming
alpha = quadprog(H, MinusOnes, [], [], Aeq, beq, lb, ub, [], options)';

% Making Sure that Not Selected alphas are zero
% (quadprog returns tiny values like 1e-9 instead of exact zeros)
alpha_AlmostZero = abs(alpha) < 1e-5;
alpha(alpha_AlmostZero) = 0;

%%% STEP 3: Find Support Vectors (observations with alpha > 0)
S = find(alpha > 0);

%%% STEP 4: Find Parameter Theta0
% For each support vector: theta0 = y_i - sum_j(alpha_j * y_j * K(x_j, x_i))
theta0 = 0;
for i = S
    theta0 = theta0 + (t(i) - MySumFunc(x(:,i), alpha(S), t(S), x(:,S), K));
end
theta0 = theta0/length(S);

% Plot Nonlinear Decision Boundary
Curve = @(x1,x2)  MySumFunc([x1; x2], alpha(S), t(S), x(:,S), K) + theta0;
CurveA = @(x1,x2) MySumFunc([x1; x2], alpha(S), t(S), x(:,S), K) + theta0 + 1;
CurveB = @(x1,x2) MySumFunc([x1; x2], alpha(S), t(S), x(:,S), K) + theta0 - 1;

PlotRange = [min(x(1,:)) max(x(1,:)) min(x(2,:)) max(x(2,:))];
figure(1)
plot(x(1,t==1), x(2,t==1), 'b*', 'DisplayName','Class 1');
hold on
plot(x(1,t==-1), x(2,t==-1), 'r+', 'DisplayName','Class 2');
plot(x(1,S), x(2,S), 'ko', 'MarkerSize', 8,'DisplayName', 'Sup. Vect.');
fimplicit(Curve, PlotRange, 'Color', 'g', ...
    'LineWidth', 2, 'DisplayName', 'Dec. Boundary', 'MeshDensity', 50);
fimplicit(CurveA, PlotRange, 'Color', 'k', ...
    'LineWidth',1,'LineStyle',':', 'DisplayName', 'Sup. Vect.', 'MeshDensity', 50);
fimplicit(CurveB, PlotRange, 'Color', 'k', ...
    'LineWidth', 1, 'LineStyle', ':', 'DisplayName', 'Sup. Vect.', 'MeshDensity', 50);
hold off
ylabel('x_2'), xlabel('x_1')
axis([-4 12 -4 11])
legend('Location', 'northwest')

% Creating proper output format for Confusion Matrix
t_org = t;
t = [t_org; t_org];
t(1,t(1,:) == -1) = 0;
t(2,t(2,:) == 1) = 0; 
t(2,t(2,:) == -1) = 1;

y = zeros(1, size(x,2));
for i = 1:size(x,2)
    y(i) = sign(MySumFunc(x(:,i), alpha(S), t_org(S), x(:,S), K) + theta0);
end
y = [y; y];
y(1,y(1,:) == 1) = 1;
y(1,y(1,:) == -1) = 0;
y(2,y(2,:) == 1) = 0;
y(2,y(2,:) == -1) = 1; 

% Plot
figure, plotconfusion(t, y, 'All Data');
title('Performance of Your KSVM Implementation')
t = t_org;

clearvars -except x t