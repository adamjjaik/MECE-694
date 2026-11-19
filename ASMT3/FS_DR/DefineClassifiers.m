function [Families, MLPFactory] = DefineClassifiers(m)
% DefineClassifiers  Classifiers and the parameter settings tried for each.
%   [Families, MLPFactory] = DefineClassifiers(m), m = PCA dimensionality.
%   Families(f).Name                  classifier name
%   Families(f).Settings(s).Label     text description of the setting
%   Families(f).Settings(s).FitPredict  handle yhat = FitPredict(Xtr, ytr, Xte)
%   Families(f).Settings(s).Params    parameters (used for the MLP + L2 step)
%   MLPFactory(Hidden, Act, LR, Epochs, Batch, L2) returns a FitPredict
%   handle for an MLP (used to build the L2-regularization step).
%
% Settings follow the trial-and-error menus in the assignment. MATLAB
% equivalents of the scikit-learn style parameters:
%   RBF SVM gamma      -> KernelScale = 1/sqrt(gamma)  (exp(-gamma*d^2) = exp(-(d/s)^2))
%   Tree max_depth d   -> MaxNumSplits = 2^d - 1 (a tree of depth d has at most 2^d - 1 splits)
%   RF max_depth       -> not limited (None); depth 15 allows up to 32,767 splits,
%                         more than the number of training samples, so it is equivalent here
%   MLP                -> Deep Learning Toolbox (trainnet) with the Adam optimizer

MLPFactory = @(H, Act, LR, Ep, BS, L2) @(Xa, ya, Xb) FitPredictMLP(Xa, ya, Xb, H, Act, LR, Ep, BS, L2);
f = 0;

%% Linear SVM: C in {0.1, 1, 10}
f = f + 1;  Families(f).Name = 'Linear SVM';  S = [];
for C = [0.1 1 10]
    T = templateSVM('KernelFunction', 'linear', 'BoxConstraint', C);
    S = AddSetting(S, sprintf('C = %g', C), @(Xa, ya, Xb) predict(fitcecoc(Xa, ya, 'Learners', T), Xb));
end
Families(f).Settings = S;

%% RBF SVM: C in {0.1, 1, 10}, gamma in {1/m, 3/m, 10/m}
f = f + 1;  Families(f).Name = 'RBF SVM';  S = [];
for C = [0.1 1 10]
    for gMult = [1 3 10]
        gamma = gMult / m;
        T = templateSVM('KernelFunction', 'gaussian', 'BoxConstraint', C, 'KernelScale', 1/sqrt(gamma));
        S = AddSetting(S, sprintf('C = %g, gamma = %d/m', C, gMult), ...
            @(Xa, ya, Xb) predict(fitcecoc(Xa, ya, 'Learners', T), Xb));
    end
end
Families(f).Settings = S;

%% k-NN: k in {3, 5, 11}, distance in {Euclidean, Manhattan}
f = f + 1;  Families(f).Name = 'k-NN';  S = [];
Dist = {'euclidean', 'Euclidean'; 'cityblock', 'Manhattan'};
for k = [3 5 11]
    for d = 1:size(Dist, 1)
        S = AddSetting(S, sprintf('k = %d, %s', k, Dist{d,2}), ...
            @(Xa, ya, Xb) predict(fitcknn(Xa, ya, 'NumNeighbors', k, 'Distance', Dist{d,1}), Xb));
    end
end
Families(f).Settings = S;

%% Decision tree: max_depth in {5, 10, 20}, min_samples_leaf in {1, 5, 10}
f = f + 1;  Families(f).Name = 'Decision Tree';  S = [];
for depth = [5 10 20]
    for leaf = [1 5 10]
        S = AddSetting(S, sprintf('max depth = %d, min leaf = %d', depth, leaf), ...
            @(Xa, ya, Xb) predict(fitctree(Xa, ya, 'MaxNumSplits', 2^depth - 1, 'MinLeafSize', leaf), Xb));
    end
end
Families(f).Settings = S;

%% Random forest: n_trees in {200, 400}, max_features in {m/3, m}, max_depth = None
f = f + 1;  Families(f).Name = 'Random Forest';  S = [];
for nT = [200 400]
    for kVar = [round(m/3) m]
        T = templateTree('NumVariablesToSample', kVar);
        S = AddSetting(S, sprintf('n trees = %d, max features = %d', nT, kVar), ...
            @(Xa, ya, Xb) predict(fitcensemble(Xa, ya, 'Method', 'Bag', ...
            'NumLearningCycles', nT, 'Learners', T), Xb));
    end
end
Families(f).Settings = S;

%% Shallow MLP: 1 hidden layer, units in {64, 128, 256}, activation in {ReLU, tanh}
% Adam, learning rate 1e-3, 50 epochs, batch size 128, no L2 (added later)
f = f + 1;  Families(f).Name = 'Shallow MLP';  S = [];
for H = [64 128 256]
    for Act = {'relu', 'tanh'}
        P = struct('Hidden', H, 'Act', Act{1}, 'LR', 1e-3, 'Epochs', 50, 'Batch', 128, 'L2', 0);
        S = AddSetting(S, MLPLabel(P), MLPFactory(P.Hidden, P.Act, P.LR, P.Epochs, P.Batch, P.L2), P);
    end
end
Families(f).Settings = S;

%% Deep MLP: 2-3 hidden layers, [128 64] or [256 128 64], activation in {ReLU, tanh}
f = f + 1;  Families(f).Name = 'Deep MLP';  S = [];
for H = {[128 64], [256 128 64]}
    for Act = {'relu', 'tanh'}
        P = struct('Hidden', H{1}, 'Act', Act{1}, 'LR', 1e-3, 'Epochs', 50, 'Batch', 128, 'L2', 0);
        S = AddSetting(S, MLPLabel(P), MLPFactory(P.Hidden, P.Act, P.LR, P.Epochs, P.Batch, P.L2), P);
    end
end
Families(f).Settings = S;
end

%% ----------------------------------------------------------------------
function S = AddSetting(S, Label, FitPredict, Params)
% Append one setting to a settings list
if nargin < 4, Params = struct(); end
New = struct('Label', Label, 'FitPredict', FitPredict, 'Params', Params);
if isempty(S), S = New; else, S(end+1) = New; end
end

function L = MLPLabel(P)
% Text description of an MLP setting
L = sprintf('[%s], %s, lr = %g, %d epochs, batch %d', ...
    strjoin(string(P.Hidden), ' '), P.Act, P.LR, P.Epochs, P.Batch);
if P.L2 > 0, L = sprintf('%s, L2 = %g', L, P.L2); end
end

function yhat = FitPredictMLP(Xtr, ytr, Xte, Hidden, Act, LR, Epochs, Batch, L2)
% Train an MLP (Adam, cross-entropy loss) and predict class labels
Ytr = categorical(ytr);
Classes = categories(Ytr);

Layers = featureInputLayer(size(Xtr, 2));
for h = Hidden
    Layers = [Layers; fullyConnectedLayer(h)];                        %#ok<AGROW>
    if strcmp(Act, 'relu'), Layers = [Layers; reluLayer];             %#ok<AGROW>
    else,                   Layers = [Layers; tanhLayer];             %#ok<AGROW>
    end
end
Layers = [Layers; fullyConnectedLayer(numel(Classes)); softmaxLayer];

% Note: trainingOptions uses L2 = 1e-4 by default, so it is set explicitly
Options = trainingOptions('adam', ...
    'InitialLearnRate', LR, 'MaxEpochs', Epochs, 'MiniBatchSize', Batch, ...
    'L2Regularization', L2, 'Shuffle', 'every-epoch', ...
    'Verbose', false, 'Plots', 'none');

Net = trainnet(Xtr, Ytr, Layers, 'crossentropy', Options);
Scores = minibatchpredict(Net, Xte);
yhat = str2double(string(scores2label(Scores, Classes)));
end
