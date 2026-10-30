%% KernelTuning.m: parameter sweeps (Table B) + kernel comparison (Table A)
clc; clear
Data = LoadData('BodyFat');

dTune        = 3;      % number of KPCA components used throughout
WeightTrain  = 0.5;
NumExecTune  = 20;     % ANN runs per setting during the sweep    (use 2 to debug)
NumExecFinal = 50;     % ANN runs for the fresh re-evaluation      (use 2 to debug)
SeedTune     = 1;      % same seed for every setting in the sweep
SeedFinal    = 2;      % different seed for the final re-evaluation
ShowSweeps   = {'Gaussian', 'Polynomial'};   % kernels shown in Table B

%% Natural scales of the data (used to centre the parameter grids)
X    = Data.Input';                  % M x p (observations x features)
Dmed = median(pdist(X));             % typical distance between observations
G    = X*X';
Gmed = median(G(:));                 % typical dot product
fprintf('Median distance = %.2f, median dot product = %.3g\n', Dmed, Gmed);

%% Grids: {Type, ParamName, Scale, Multipliers}  ->  parameter = Scale * Multipliers
Cfg = { ...
  'Linear',            'exponent', 1,       1;                       % 1 = true linear (= PCA)
  'Polynomial',        'degree',   1,       [2 3 4];
  'Gaussian',          'sigma',    Dmed,    [0.1 0.3 1 3 10];
  'HypTan',            'alpha',    1/Gmed,  [0.01 0.1 1 10];
  'Laplacian',         'sigma',    Dmed,    [0.1 0.3 1 3 10];
  'RationalQuadratic', 'c',        Dmed^2,  [0.01 0.1 1 10 100];
  'MultiQuadric',      'c',        Dmed,    [0.1 0.3 1 3 10];
  'Power',             'd',        1,       [0.5 1 1.5 2];
  'Log',               'd',        1,       [0.5 1 1.5 2] };
nK = size(Cfg, 1);

%% Step 1: sweep each kernel's parameter (raw data for Table B)
SweepParam = cell(nK, 1);
SweepErr   = cell(nK, 1);
for k = 1:nK
    SweepParam{k} = Cfg{k,3} * Cfg{k,4};
    SweepErr{k}   = zeros(size(SweepParam{k}));
    for i = 1:numel(SweepParam{k})
        SweepErr{k}(i) = EvalKPCA(Data, Cfg{k,1}, SweepParam{k}(i), dTune, ...
                                  WeightTrain, NumExecTune, SeedTune);
    end
    fprintf('%-18s swept (%d values)\n', Cfg{k,1}, numel(SweepParam{k}));
end

%% PCA reference under both protocols
[~, score] = pca(Data.Input');
Dpca = Data;  Dpca.Input = score(:, 1:dTune)';  Dpca.d = dTune;
rng(SeedTune);  Err_PCA_Tune  = FeatureSelectionObjective(ones(1,dTune), Dpca, WeightTrain, NumExecTune);
rng(SeedFinal); Err_PCA_Final = FeatureSelectionObjective(ones(1,dTune), Dpca, WeightTrain, NumExecFinal);

%% Step 2: best parameter for each kernel
BestParam = zeros(nK, 1);  TuneErr = zeros(nK, 1);  AtEdge = false(nK, 1);
for k = 1:nK
    [TuneErr(k), j] = min(SweepErr{k});          % NaN (failed) values are ignored
    BestParam(k)    = SweepParam{k}(j);
    AtEdge(k)       = numel(SweepParam{k}) > 1 && (j == 1 || j == numel(SweepParam{k}));
end

%% Step 3: fresh re-evaluation of the winners (new seed, more runs)
% Taking the minimum of several noisy numbers biases it low, so TuneErr is
% optimistic; FinalErr is the honest value to report.
FinalErr = zeros(nK, 1);
for k = 1:nK
    FinalErr(k) = EvalKPCA(Data, Cfg{k,1}, BestParam(k), dTune, ...
                           WeightTrain, NumExecFinal, SeedFinal);
end

%% Table A: kernel comparison at each kernel's best parameter
Kernel    = Cfg(:,1);
ParamName = Cfg(:,2);
BestMult  = BestParam ./ cell2mat(Cfg(:,3));     % best parameter as a multiple of the data scale
TableA = table(Kernel, ParamName, BestParam, BestMult, AtEdge, TuneErr, FinalErr);
TableA = sortrows(TableA, 'FinalErr');
fprintf('\nTable A: kernel comparison at best parameter (d = %d)\n', dTune);
disp(TableA)
fprintf('PCA reference: tuning protocol %.4f, final protocol %.4f\n', Err_PCA_Tune, Err_PCA_Final);

%% Table B: effect of the parameter, for the kernels in ShowSweeps
figure(11); clf
for s = 1:numel(ShowSweeps)
    k = find(strcmp(Cfg(:,1), ShowSweeps{s}));
    TableB = table(SweepParam{k}(:), Cfg{k,4}(:), SweepErr{k}(:), ...
        'VariableNames', {Cfg{k,2}, 'MultipleOfScale', 'Error'});
    fprintf('\nTable B: %s kernel, effect of %s (d = %d)\n', ShowSweeps{s}, Cfg{k,2}, dTune);
    disp(TableB)

    subplot(1, numel(ShowSweeps), s)
    plot(Cfg{k,4}, SweepErr{k}, '-o'); grid on
    if max(Cfg{k,4}) / min(Cfg{k,4}) > 20, set(gca, 'XScale', 'log'); end
    yline(Err_PCA_Tune, '--', 'PCA');
    xlabel([Cfg{k,2} ' (multiple of data scale)']);
    ylabel('Weighted train + test error');
    title([ShowSweeps{s} ' kernel']);
end

save('KernelTuning_Results.mat', 'Cfg', 'SweepParam', 'SweepErr', 'TableA', ...
     'Err_PCA_Tune', 'Err_PCA_Final', 'dTune');

%% Helper (local functions must stay at the END of the file)
function err = EvalKPCA(Data, Type, Param, d, WeightTrain, nExec, seed)
% Error of the ANN on the d-component KPCA embedding; NaN if the kernel fails
    Kt.Type = Type;  Kt.Parameter = Param;
    rng(seed);        % same seed for every setting = same ANN inits and splits,
                      % so differences between settings are not just noise
    try
        Z = KernelPCA_Incomplete(Data.Input', Kt, d);
        if ~isreal(Z) || any(~isfinite(Z(:)))
            error('embedding is complex or contains Inf/NaN');
        end
        Dt = Data;  Dt.Input = Z;  Dt.d = d;
        err = FeatureSelectionObjective(ones(1, d), Dt, WeightTrain, nExec);
    catch ME
        warning('%s (param %g) failed: %s', Type, Param, ME.message);
        err = NaN;
    end
end