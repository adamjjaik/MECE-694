%% Q2_ComparisonTable.m
% MECE 694 - Assignment 3, Question 2
% Compares the accuracy of:
%   (1) Your MLP (backpropagation + fminunc, sigmoid activations)
%   (2) MATLAB MLP with trainlm
%   (3) MATLAB MLP with trainbr
%   (4) MATLAB MLP with trainscg
% for different numbers of hidden-layer neurons. Each scenario is repeated
% nRuns times (new random dataset + new random initial weights each run),
% and the mean +/- standard deviation of the accuracy is reported.
%
% Accuracy is computed on all data (resubstitution), same as the
% confusion plots in Classification_ANN_Incomplete.
%
% Requires (in the same folder): RandomInitialWeights_Incomplete.m,
% MyANNObjecFunction_Incomplete.m, PredictMLPOutputs.m

clear
clc
close all

%% Settings
hiddenSizes = [2 5 10 15 20];                       % neurons in hidden layer
nRuns       = 10;                                   % repetitions per scenario
methodNames = {'Your MLP (Sigmoid)', 'MATLAB MLP (trainlm)', ...
               'MATLAB MLP (trainbr)', 'MATLAB MLP (trainscg)'};
trainFcns   = {'trainlm', 'trainbr', 'trainscg'};

NumClasses = 2;
lambda     = 0;                                     % regularization weight (same as template)

% Options for fminunc (same as template, but no iteration display)
options = optimoptions('fminunc', ...
    'Algorithm', 'trust-region', ...
    'SpecifyObjectiveGradient', true, ...
    'Display', 'off', ...
    'MaxIterations', 150);

% Storage: accuracy(method, hiddenSize, run)
accuracy = zeros(numel(methodNames), numel(hiddenSizes), nRuns);

tStart = tic;

%% Main loop
for h = 1:numel(hiddenSizes)
    hiddenLayerSize = hiddenSizes(h);

    for r = 1:nRuns
        fprintf('Hidden neurons = %2d | Run %2d of %d ... ', hiddenLayerSize, r, nRuns);

        % ---------- (1) Create a new random dataset (same as Section 1) ----------
        Mu1 = [5 5];  Sigma1 = [3 0.1; 0.1 3];
        X1  = mvnrnd(Mu1, Sigma1, 50)';
        Mu1 = [6 -2]; Sigma1 = [2 0; 0 2];
        X1  = [X1 mvnrnd(Mu1, Sigma1, 50)'];

        Mu2 = [1 1];  Sigma2 = [5 0.5; 0.5 5];
        X2  = mvnrnd(Mu2, Sigma2, 100)';

        x     = [X1'; X2']';
        t_org = [ones(size(X1,2),1); -1*ones(size(X2,2),1)]';

        % One-hot target format: row 1 = Class 1, row 2 = Class 2
        t = [t_org; t_org];
        t(1, t(1,:) == -1) = 0;
        t(2, t(2,:) == 1) = 0;  t(2, t(2,:) == -1) = 1;

        % True class index (1 or 2) for each observation
        [~, trueClass] = max(t, [], 1);

        NumFeatures = size(x, 1);

        % ---------- (2) Your MLP: backpropagation + fminunc ----------
        Initial_theta1 = RandomInitialWeights_Incomplete(NumFeatures, hiddenLayerSize);
        Initial_theta2 = RandomInitialWeights_Incomplete(hiddenLayerSize, NumClasses);
        ANN_Initial_Weights = [Initial_theta1(:); Initial_theta2(:)];

        Objective = @(ANN_Weights) MyANNObjecFunction_Incomplete(ANN_Weights, ...
            NumFeatures, hiddenLayerSize, NumClasses, x, t, lambda);

        ANN_Weights = fminunc(Objective, ANN_Initial_Weights, options);

        theta1 = reshape(ANN_Weights(1:hiddenLayerSize * (NumFeatures + 1)), ...
            hiddenLayerSize, (NumFeatures + 1));
        theta2 = reshape(ANN_Weights((1 + (hiddenLayerSize * (NumFeatures + 1))):end), ...
            NumClasses, (hiddenLayerSize + 1));

        y = PredictMLPOutputs(theta1, theta2, x);       % one-hot predictions
        [~, predClass] = max(y, [], 1);
        accuracy(1, h, r) = mean(predClass == trueClass);

        % ---------- (3) MATLAB MLPs (same setup as Section 2) ----------
        for k = 1:numel(trainFcns)
            net = fitnet(hiddenLayerSize, trainFcns{k});

            net.input.processFcns  = {'removeconstantrows', 'mapminmax'};
            net.output.processFcns = {'removeconstantrows', 'mapminmax'};

            net.divideFcn = 'dividerand';
            net.divideMode = 'sample';
            net.divideParam.trainRatio = 100/100;
            net.divideParam.valRatio   = 0/100;
            net.divideParam.testRatio  = 0/100;

            net.performFcn = 'mse';

            net.trainParam.showWindow      = false;     % no training GUI
            net.trainParam.showCommandLine = false;
            net.trainParam.epochs          = 1000;
            net.trainParam.goal            = 0;

            net = train(net, x, t);
            yNet = net(x);

            % Predicted class = output neuron with the largest value
            % (same rule plotconfusion uses)
            [~, predClass] = max(yNet, [], 1);
            accuracy(k + 1, h, r) = mean(predClass == trueClass);
        end

        fprintf('done (%.1f s elapsed)\n', toc(tStart));
    end
end

%% Results: mean and standard deviation (in %)
accMean = 100 * mean(accuracy, 3);     % methods x hiddenSizes
accStd  = 100 * std(accuracy, 0, 3);

% Print the table in the same layout as the assignment
fprintf('\n===== Question 2: Accuracy (%%), mean +/- std over %d runs =====\n', nRuns);
fprintf('%-24s', 'Hidden neurons');
fprintf('%16d', hiddenSizes);
fprintf('\n');
for m = 1:numel(methodNames)
    fprintf('%-24s', methodNames{m});
    for h = 1:numel(hiddenSizes)
        fprintf('%16s', sprintf('%.1f +/- %.1f', accMean(m,h), accStd(m,h)));
    end
    fprintf('\n');
end

% Also store as a MATLAB table (easy to copy into the report)
colNames = arrayfun(@(n) sprintf('H_%d', n), hiddenSizes, 'UniformOutput', false);
cellResults = arrayfun(@(mu, sd) sprintf('%.1f ± %.1f', mu, sd), accMean, accStd, ...
    'UniformOutput', false);
ResultsTable = cell2table(cellResults, 'VariableNames', colNames, 'RowNames', methodNames);
disp(ResultsTable)

% Save everything so the runs don't have to be repeated
save('Q2_Results.mat', 'accuracy', 'accMean', 'accStd', 'hiddenSizes', ...
    'methodNames', 'ResultsTable');

%% Optional figure: mean accuracy vs hidden neurons (with std error bars)
figure
hold on
markers = {'o-', 's-', '^-', 'd-'};
for m = 1:numel(methodNames)
    errorbar(hiddenSizes, accMean(m,:), accStd(m,:), markers{m}, ...
        'LineWidth', 1.5, 'DisplayName', methodNames{m});
end
hold off
grid on
xlabel('Number of neurons in the hidden layer')
ylabel('Accuracy (%)')
title(sprintf('MLP accuracy (mean \\pm std over %d runs)', nRuns))
legend('Location', 'southeast')
xticks(hiddenSizes)
