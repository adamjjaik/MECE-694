%% Q4_KernelTuning.m
% MECE 694 - Assignment 3, Question 4 (parameter tuning)
% Tunes the kernel parameter of your KSVM for the Gaussian, Logistic and
% Hyperbolic Tangent kernels over several random problems, then plots
% accuracy vs. parameter value and marks the selected parameter.
%
% Each random problem has a TRAINING set (used to fit the KSVM) and a
% separate TEST set drawn from the same distributions. Parameters are
% selected by TEST accuracy: training (resubstitution) accuracy always
% favours very flexible boundaries that memorize the data.
%
% The Gaussian sweep also answers the last Question 4 item (effect of the
% Gaussian kernel parameter).
%
% Requires: MyKSVM.m, KernelFunction_Incomplete.m, MySumFunc.m

clear
clc
close all
rng(0)                      % fixed seed so the results are repeatable

%% Settings
nProblems = 5;              % random problems used for tuning
C = 10;                     % box constraint (same as Section 3)

sigmaGrid = [0.25 0.5 0.75 1 1.5 2 3 4 6 8];          % Gaussian sigma
beta1Grid = [0.001 0.003 0.01 0.03 0.1 0.3 1];         % Logistic/HypTan slope
beta0Grid = [-1 0 1];                                  % Logistic/HypTan offset

%% Create the random problems (training + test set for each)
Data = struct('xTr', {}, 'tTr', {}, 'xTe', {}, 'tTe', {});
for p = 1:nProblems
    [Data(p).xTr, Data(p).tTr] = GenerateData();
    [Data(p).xTe, Data(p).tTe] = GenerateData();
end

tStart = tic;

%% Gaussian kernel: sweep sigma
nS = numel(sigmaGrid);
G_tr = zeros(nS, nProblems); G_te = G_tr; G_nSV = G_tr;
for s = 1:nS
    Kernel.Type = 'Gaussian';
    Kernel.Parameter = sigmaGrid(s);
    for p = 1:nProblems
        [G_tr(s,p), G_te(s,p), G_nSV(s,p)] = MyKSVM(Data(p).xTr, Data(p).tTr, ...
            Data(p).xTe, Data(p).tTe, Kernel, C);
    end
    fprintf('Gaussian  sigma = %-5g done (%.0f s)\n', sigmaGrid(s), toc(tStart));
end

%% Logistic and Hyperbolic Tangent kernels: sweep [beta0 beta1]
nB0 = numel(beta0Grid); nB1 = numel(beta1Grid);
L_tr = zeros(nB0, nB1, nProblems); L_te = L_tr; L_psd = false(size(L_tr));
T_tr = L_tr; T_te = L_tr; T_psd = L_psd;
for b0 = 1:nB0
    for b1 = 1:nB1
        for p = 1:nProblems
            Kernel.Parameter = [beta0Grid(b0) beta1Grid(b1)];

            Kernel.Type = 'Logistic';
            [L_tr(b0,b1,p), L_te(b0,b1,p), ~, L_psd(b0,b1,p)] = MyKSVM(Data(p).xTr, ...
                Data(p).tTr, Data(p).xTe, Data(p).tTe, Kernel, C);

            Kernel.Type = 'HypTan';
            [T_tr(b0,b1,p), T_te(b0,b1,p), ~, T_psd(b0,b1,p)] = MyKSVM(Data(p).xTr, ...
                Data(p).tTr, Data(p).xTe, Data(p).tTe, Kernel, C);
        end
        fprintf('Logistic/HypTan  beta0 = %-3g beta1 = %-6g done (%.0f s)\n', ...
            beta0Grid(b0), beta1Grid(b1), toc(tStart));
    end
end

%% Mean and std over problems (in %)
G_teMean = 100*mean(G_te, 2, 'omitnan');  G_teStd = 100*std(G_te, 0, 2, 'omitnan');
G_trMean = 100*mean(G_tr, 2, 'omitnan');
L_teMean = 100*mean(L_te, 3, 'omitnan');  L_teStd = 100*std(L_te, 0, 3, 'omitnan');
T_teMean = 100*mean(T_te, 3, 'omitnan');  T_teStd = 100*std(T_te, 0, 3, 'omitnan');

%% Select the best parameters (highest mean TEST accuracy)
[~, iG] = max(G_teMean);
bestSigma = sigmaGrid(iG);

[~, lin] = max(L_teMean(:));
[iL0, iL1] = ind2sub(size(L_teMean), lin);
bestLogistic = [beta0Grid(iL0) beta1Grid(iL1)];

[~, lin] = max(T_teMean(:));
[iT0, iT1] = ind2sub(size(T_teMean), lin);
bestHypTan = [beta0Grid(iT0) beta1Grid(iT1)];

fprintf('\n===== Selected parameters (C = %g, %d random problems) =====\n', C, nProblems);
fprintf('Gaussian : sigma = %g                -> test acc = %.1f +/- %.1f %%\n', ...
    bestSigma, G_teMean(iG), G_teStd(iG));
fprintf('Logistic : [beta0 beta1] = [%g %g]   -> test acc = %.1f +/- %.1f %%\n', ...
    bestLogistic, L_teMean(iL0,iL1), L_teStd(iL0,iL1));
fprintf('HypTan   : [beta0 beta1] = [%g %g]   -> test acc = %.1f +/- %.1f %%\n', ...
    bestHypTan, T_teMean(iT0,iT1), T_teStd(iT0,iT1));
fprintf('H needed a PSD correction in %d of %d Logistic runs and %d of %d HypTan runs\n', ...
    nnz(L_psd), numel(L_psd), nnz(T_psd), numel(T_psd));

save('Q4_Tuning.mat', 'sigmaGrid', 'beta0Grid', 'beta1Grid', 'C', 'nProblems', ...
    'G_tr', 'G_te', 'G_nSV', 'L_tr', 'L_te', 'L_psd', 'T_tr', 'T_te', 'T_psd', ...
    'bestSigma', 'bestLogistic', 'bestHypTan');

%% Figure 1: tuning results with the selected parameter marked
figure('Position', [100 100 1400 420])
colors = lines(nB0);

% --- Gaussian ---
subplot(1,3,1)
errorbar(sigmaGrid, G_teMean, G_teStd, 'o-', 'LineWidth', 1.5, ...
    'DisplayName', 'Test (new data)');
hold on
plot(sigmaGrid, G_trMean, 's--', 'LineWidth', 1.2, 'DisplayName', 'Training (resubstitution)');
plot(bestSigma, G_teMean(iG), 'kp', 'MarkerSize', 16, 'MarkerFaceColor', 'y', ...
    'DisplayName', sprintf('Selected: \\sigma = %g', bestSigma));
hold off
set(gca, 'XScale', 'log')
grid on
xlabel('\sigma')
ylabel('Accuracy (%)')
title('Gaussian kernel')
legend('Location', 'southwest')

% --- Logistic ---
subplot(1,3,2)
hold on
for b0 = 1:nB0
    errorbar(beta1Grid, L_teMean(b0,:), L_teStd(b0,:), 'o-', 'Color', colors(b0,:), ...
        'LineWidth', 1.5, 'DisplayName', sprintf('\\beta_0 = %g', beta0Grid(b0)));
end
plot(bestLogistic(2), L_teMean(iL0,iL1), 'kp', 'MarkerSize', 16, 'MarkerFaceColor', 'y', ...
    'DisplayName', sprintf('Selected: [%g, %g]', bestLogistic));
hold off
set(gca, 'XScale', 'log')
grid on
xlabel('\beta_1')
ylabel('Test accuracy (%)')
title('Logistic kernel')
legend('Location', 'southwest')

% --- Hyperbolic Tangent ---
subplot(1,3,3)
hold on
for b0 = 1:nB0
    errorbar(beta1Grid, T_teMean(b0,:), T_teStd(b0,:), 'o-', 'Color', colors(b0,:), ...
        'LineWidth', 1.5, 'DisplayName', sprintf('\\beta_0 = %g', beta0Grid(b0)));
end
plot(bestHypTan(2), T_teMean(iT0,iT1), 'kp', 'MarkerSize', 16, 'MarkerFaceColor', 'y', ...
    'DisplayName', sprintf('Selected: [%g, %g]', bestHypTan));
hold off
set(gca, 'XScale', 'log')
grid on
xlabel('\beta_1')
ylabel('Test accuracy (%)')
title('Hyperbolic tangent kernel')
legend('Location', 'southwest')

sgtitle(sprintf('KSVM parameter tuning (C = %g, mean \\pm std over %d random problems)', ...
    C, nProblems))

%% Figure 2: Gaussian - number of support vectors vs sigma (for discussion)
figure
plot(sigmaGrid, mean(G_nSV, 2), 'o-', 'LineWidth', 1.5)
set(gca, 'XScale', 'log')
grid on
xlabel('\sigma')
ylabel('Number of support vectors (of 200)')
title('Gaussian kernel: support vectors vs \sigma')

%% Local function: create one random problem (same as Section 1)
function [x, t] = GenerateData()
Mu1 = [5 5];  Sigma1 = [3 0.1; 0.1 3];
X1  = mvnrnd(Mu1, Sigma1, 50)';
Mu1 = [6 -2]; Sigma1 = [2 0; 0 2];
X1  = [X1 mvnrnd(Mu1, Sigma1, 50)'];

Mu2 = [1 1];  Sigma2 = [5 0.5; 0.5 5];
X2  = mvnrnd(Mu2, Sigma2, 100)';

x = [X1 X2];
t = [ones(1, size(X1,2)) -ones(1, size(X2,2))];
end
