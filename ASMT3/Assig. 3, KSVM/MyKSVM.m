function [accTrain, accTest, nSV, psdFixed, Model] = MyKSVM(xTrain, tTrain, xTest, tTest, Kernel, C)
% MyKSVM - Function version of the KSVM in Section 3 of
% "Classification_KSVM_Incomplete" (same steps, no plots), so it can be
% called in a loop for parameter tuning and the Question 4 comparison.
%
% Inputs:
%   xTrain, tTrain : training data (p x M) and labels (1 x M, values 1/-1)
%   xTest,  tTest  : separate test data and labels (same format)
%   Kernel         : structure with fields Type and Parameter
%   C              : box constraint
% Outputs:
%   accTrain : accuracy on the training data (resubstitution)
%   accTest  : accuracy on the separate test data
%   nSV      : number of support vectors
%   psdFixed : true if H was not positive semi-definite and had to be
%              corrected (happens for Logistic/HypTan, which are not
%              valid Mercer kernels for all parameter values)
%   Model    : trained model (alphaS, tS, xS, K, theta0), used to plot the
%              decision boundary
%
% Requires: KernelFunction_Incomplete.m, MySumFunc.m

K = KernelFunction_Incomplete(Kernel);
M = size(xTrain, 2);
Model = struct('alphaS', [], 'tS', [], 'xS', [], 'K', K, 'theta0', NaN);

%%% STEP 1: Matrix H, h_ij = y_i * y_j * K(x_i, x_j)
H = zeros(M, M);
for i = 1:M
    for j = i:M
        H(i,j) = tTrain(i) * tTrain(j) * K(xTrain(:,i), xTrain(:,j));
        H(j,i) = H(i,j);
    end
end

% quadprog (interior-point-convex) needs H to be positive semi-definite.
% Gaussian always gives a PSD matrix; Logistic/HypTan may not. If H has
% significantly negative eigenvalues, replace it with the nearest PSD
% matrix (set negative eigenvalues to zero).
H = (H + H') / 2;
[V, D] = eig(H);
d = diag(D);
psdFixed = false;
if min(d) < -1e-6 * max(abs(d))
    d(d < 0) = 0;
    H = V * diag(d) * V';
    H = (H + H') / 2;
    psdFixed = true;
end

%%% STEP 2: Solve the quadratic programming problem
options = optimoptions('quadprog', 'Algorithm', 'interior-point-convex', ...
    'Display', 'off', 'MaxIterations', 100);
alpha = quadprog(H, -ones(M,1), [], [], tTrain, 0, ...
    zeros(M,1), C*ones(M,1), [], options)';

if isempty(alpha)
    accTrain = NaN; accTest = NaN; nSV = 0;
    return
end
alpha(abs(alpha) < 1e-5) = 0;

%%% STEP 3: Support vectors
S = find(alpha > 0);
nSV = numel(S);
if nSV == 0
    accTrain = NaN; accTest = NaN;
    return
end

%%% STEP 4: theta0 (averaged over support vectors, as in Section 3)
theta0 = 0;
for i = S
    theta0 = theta0 + (tTrain(i) - MySumFunc(xTrain(:,i), alpha(S), tTrain(S), xTrain(:,S), K));
end
theta0 = theta0 / nSV;

Model.alphaS = alpha(S);
Model.tS     = tTrain(S);
Model.xS     = xTrain(:,S);
Model.theta0 = theta0;

%%% Accuracy on training and test data
yTrain = KSVMPredict(xTrain, alpha(S), tTrain(S), xTrain(:,S), K, theta0);
yTest  = KSVMPredict(xTest,  alpha(S), tTrain(S), xTrain(:,S), K, theta0);
accTrain = mean(yTrain == tTrain);
accTest  = mean(yTest  == tTest);
end

function y = KSVMPredict(xNew, alphaS, tS, xS, K, theta0)
% Class (1 or -1) for each column of xNew
y = zeros(1, size(xNew, 2));
for i = 1:size(xNew, 2)
    f = MySumFunc(xNew(:,i), alphaS, tS, xS, K) + theta0;
    if f >= 0
        y(i) = 1;
    else
        y(i) = -1;
    end
end
end