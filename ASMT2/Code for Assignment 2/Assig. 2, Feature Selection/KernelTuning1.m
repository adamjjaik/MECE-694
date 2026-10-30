%% KernelTable.m: KPCA error for every kernel at small/medium/large parameter (d = 3)
clc; clear
Data = LoadData('BodyFat');
d = 3;  WeightTrain = 0.5;  NumExec = 50;  Seed = 1;   % use NumExec = 2 to debug

X    = Data.Input';
Dmed = median(pdist(X));           % typical distance between observations
G    = X*X';  Gmed = median(G(:)); % typical dot product

% {Display name, Type, [small medium large] parameter values}
Rows = { ...
  'Linear (exponent)',        'Linear',            [1 2 3];             % 1 = true linear (= PCA)
  'Polynomial (degree)',      'Polynomial',        [2 3 4];
  'Gaussian (sigma)',         'Gaussian',          Dmed*[0.1 1 10];
  'HypTan (alpha)',           'HypTan',            (1/Gmed)*[0.1 1 10];
  'Laplacian (sigma)',        'Laplacian',         Dmed*[0.1 1 10];
  'RationalQuadratic (c)',    'RationalQuadratic', Dmed^2*[0.1 1 10];
  'MultiQuadric (c)',         'MultiQuadric',      Dmed*[0.1 1 10];
  'Power (d)',                'Power',             [0.5 1 2];           % d = 2 equals linear after centring
  'Log (d)',                  'Log',               [0.5 1 2] };

% Error for every kernel/parameter cell
nR  = size(Rows,1);
Err = zeros(nR, 3);
for r = 1:nR
    for c = 1:3
        Err(r,c) = EvalKPCA(Data, Rows{r,2}, Rows{r,3}(c), d, WeightTrain, NumExec, Seed);
    end
    fprintf('%s done\n', Rows{r,1});
end

% Noise estimate: same setting (Linear, exponent 1) repeated with 5 different seeds
NoiseRuns = arrayfun(@(s) EvalKPCA(Data, 'Linear', 1, d, WeightTrain, NumExec, s), 11:15);
NoiseStd  = std(NoiseRuns);

% Table of errors, plus a table of the actual parameter values used
T = array2table(Err, 'RowNames', Rows(:,1), 'VariableNames', {'Small','Medium','Large'});
P = array2table(vertcat(Rows{:,3}), 'RowNames', Rows(:,1), 'VariableNames', {'Small','Medium','Large'});

fprintf('\nKPCA error (d = %d), %d ANN runs per cell\n', d, NumExec);
disp(T)
fprintf('Parameter values used:\n');
disp(P)
fprintf('Run-to-run noise (Linear, 5 seeds): %.2f +/- %.2f\n', mean(NoiseRuns), NoiseStd);
fprintf('Scales: D (median distance) = %.2f, G (median dot product) = %.3g\n', Dmed, Gmed);

save('KernelTable_Results.mat', 'T', 'P', 'Rows', 'Err', 'NoiseRuns', 'Dmed', 'Gmed');

%% Helper (must stay at the end of the file)
function err = EvalKPCA(Data, Type, Param, d, WeightTrain, nExec, seed)
    Kt.Type = Type;  Kt.Parameter = Param;
    rng(seed);
    try
        Z = KernelPCA_Incomplete(Data.Input', Kt, d);
        if ~isreal(Z) || any(~isfinite(Z(:))), error('bad embedding'); end
        Dt = Data;  Dt.Input = Z;  Dt.d = d;
        err = FeatureSelectionObjective(ones(1,d), Dt, WeightTrain, nExec);
    catch
        err = NaN;   % kernel failed numerically
    end
end