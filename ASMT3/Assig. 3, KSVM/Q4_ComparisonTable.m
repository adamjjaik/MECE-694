%% Q4_ComparisonTable.m
% MECE 694 - Assignment 3, Question 4 (comparison table)
% Compares your KSVM (Gaussian, Logistic, Hyperbolic Tangent kernels with
% the parameters selected in Q4_KernelTuning) with MATLAB's KSVM
% (Gaussian and polynomial kernels) on 3 NEW random problems.
%
% Accuracy is reported as in the assignment (resubstitution, i.e. the
% accuracy from the confusion matrix on the training data). Accuracy on a
% separate test set from the same distributions is also reported.
%
% Requires: MyKSVM.m, KernelFunction_Incomplete.m, MySumFunc.m,
%           Q4_Tuning.mat (from Q4_KernelTuning)

clear
clc
close all
rng(1)                      % different seed from tuning -> new problems

%% Settings
nProblems = 3;
C = 10;                     % same box constraint for every model

% Selected parameters from tuning (edit here to override, e.g. for
% robustness use bestLogistic = [-1 0.01])
load('Q4_Tuning.mat', 'bestSigma', 'bestLogistic', 'bestHypTan');

YourKernels(1).Type = 'Gaussian'; YourKernels(1).Parameter = bestSigma;
YourKernels(2).Type = 'Logistic'; YourKernels(2).Parameter = bestLogistic;
YourKernels(3).Type = 'HypTan';   YourKernels(3).Parameter = bestHypTan;

rowNames = {sprintf('Your KSVM (Gaussian, sigma = %g)', bestSigma), ...
            sprintf('Your KSVM (Logistic, beta = [%g %g])', bestLogistic), ...
            sprintf('Your KSVM (HypTan, beta = [%g %g])', bestHypTan), ...
            'MATLAB KSVM (Gaussian)', ...
            'MATLAB KSVM (Polynomial, order 3)'};

accTrain = zeros(5, nProblems);
accTest  = zeros(5, nProblems);
Models   = cell(3, nProblems);
Problems = struct('xTr', {}, 'tTr', {});

%% Run the comparison
for p = 1:nProblems
    [xTr, tTr] = GenerateData();
    [xTe, tTe] = GenerateData();
    Problems(p).xTr = xTr; Problems(p).tTr = tTr;

    % --- Your KSVM with the three kernels ---
    for k = 1:3
        [accTrain(k,p), accTest(k,p), ~, ~, Models{k,p}] = ...
            MyKSVM(xTr, tTr, xTe, tTe, YourKernels(k), C);
    end

    % --- MATLAB KSVM, Gaussian ---
    % MATLAB's Gaussian kernel is exp(-||xi - xj||^2 / s^2) with s = KernelScale,
    % so s = sqrt(2)*sigma gives exactly the same kernel as yours.
    SVM = fitcsvm(xTr', tTr', 'BoxConstraint', C, ...
        'KernelFunction', 'gaussian', 'KernelScale', sqrt(2)*bestSigma, ...
        'Solver', 'ISDA');
    accTrain(4,p) = mean(predict(SVM, xTr')' == tTr);
    accTest(4,p)  = mean(predict(SVM, xTe')' == tTe);

    % --- MATLAB KSVM, polynomial (order 3) ---
    % KernelScale 'auto' rescales the inputs; without it (1 + xi'xj)^3 reaches
    % ~10^6 on this data and the solver struggles.
    SVM = fitcsvm(xTr', tTr', 'BoxConstraint', C, ...
        'KernelFunction', 'polynomial', 'PolynomialOrder', 3, ...
        'KernelScale', 'auto', 'Solver', 'ISDA');
    accTrain(5,p) = mean(predict(SVM, xTr')' == tTr);
    accTest(5,p)  = mean(predict(SVM, xTe')' == tTe);

    fprintf('Random problem %d done\n', p);
end

%% Print the tables (in %)
PrintTable('Accuracy from confusion matrix (resubstitution), %', rowNames, 100*accTrain);
PrintTable('Accuracy on a separate test set, %', rowNames, 100*accTest);

% LaTeX rows for the report table (resubstitution accuracy)
fprintf('\n%% ---- LaTeX rows (resubstitution accuracy, %%) ----\n');
for r = 1:numel(rowNames)
    fprintf('        %-40s', strrep(rowNames{r}, 'sigma', '$\sigma$'));
    fprintf(' & %.1f', 100*accTrain(r,:));
    fprintf(' \\\\\n');
end

save('Q4_Results.mat', 'accTrain', 'accTest', 'rowNames', 'C', 'YourKernels');

%% Figure: decision boundaries of your KSVM for random problem 1
figure('Position', [100 100 1400 420])
for k = 1:3
    subplot(1,3,k)
    PlotBoundary(Problems(1).xTr, Problems(1).tTr, Models{k,1});
    title(sprintf('%s (acc = %.1f%%)', rowNames{k}, 100*accTrain(k,1)), 'FontSize', 9)
end
sgtitle(sprintf('Your KSVM decision boundaries, random problem 1 (C = %g)', C))

%% Local functions
function [x, t] = GenerateData()
% One random problem (same as Section 1)
Mu1 = [5 5];  Sigma1 = [3 0.1; 0.1 3];
X1  = mvnrnd(Mu1, Sigma1, 50)';
Mu1 = [6 -2]; Sigma1 = [2 0; 0 2];
X1  = [X1 mvnrnd(Mu1, Sigma1, 50)'];
Mu2 = [1 1];  Sigma2 = [5 0.5; 0.5 5];
X2  = mvnrnd(Mu2, Sigma2, 100)';
x = [X1 X2];
t = [ones(1, size(X1,2)) -ones(1, size(X2,2))];
end

function PrintTable(heading, rowNames, A)
fprintf('\n===== %s =====\n', heading);
fprintf('%-42s%12s%12s%12s%12s\n', 'Model', 'Problem 1', 'Problem 2', 'Problem 3', 'Mean');
for r = 1:numel(rowNames)
    fprintf('%-42s', rowNames{r});
    fprintf('%12.1f', A(r,:));
    fprintf('%12.1f\n', mean(A(r,:)));
end
end

function PlotBoundary(x, t, Model)
% Data, support vectors, decision boundary (f = 0) and margins (f = +/-1)
f = @(x1, x2) MySumFunc([x1; x2], Model.alphaS, Model.tS, Model.xS, Model.K) + Model.theta0;
Range = [min(x(1,:)) max(x(1,:)) min(x(2,:)) max(x(2,:))];
plot(x(1,t==1), x(2,t==1), 'b*', 'DisplayName', 'Class 1');
hold on
plot(x(1,t==-1), x(2,t==-1), 'r+', 'DisplayName', 'Class 2');
plot(Model.xS(1,:), Model.xS(2,:), 'ko', 'MarkerSize', 7, 'DisplayName', 'Sup. Vect.');
fimplicit(f, Range, 'Color', 'g', 'LineWidth', 2, 'DisplayName', 'Dec. Boundary', 'MeshDensity', 50);
fimplicit(@(a,b) f(a,b) - 1, Range, 'Color', 'k', 'LineStyle', ':', 'DisplayName', 'Margin', 'MeshDensity', 50);
fimplicit(@(a,b) f(a,b) + 1, Range, 'Color', 'k', 'LineStyle', ':', 'HandleVisibility', 'off', 'MeshDensity', 50);
hold off
axis([-4 12 -4 11])
xlabel('x_1'), ylabel('x_2')
legend('Location', 'northwest', 'FontSize', 7)
end
