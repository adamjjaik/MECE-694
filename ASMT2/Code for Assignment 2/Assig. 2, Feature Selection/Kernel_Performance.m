%% KernelFigures.m: parameter-effect plots + best-kernel comparison (d = 3)
clc; clear
Data = LoadData('BodyFat');
load('KernelTable_Results.mat');   % T, Rows, Err, NoiseRuns, Dmed, Gmed
d = 3;  WeightTrain = 0.5;
RefMean = mean(NoiseRuns);  RefStd = std(NoiseRuns);   % linear kernel (= PCA) over 5 seeds

%% Figure A: error vs. kernel parameter
ScaleRows = {'Gaussian','HypTan','Laplacian','RationalQuadratic','MultiQuadric'};
ExpRows   = {'Linear','Polynomial','Power','Log'};
ExpVals   = {[1 2 3], [2 3 4], [0.5 1 2], [0.5 1 2]};

figure(20); clf
subplot(1,2,1); hold on
for k = 1:numel(ScaleRows)
    r = strcmp(Rows(:,2), ScaleRows{k});
    plot([0.1 1 10], Err(r,:), '-o', 'LineWidth', 1.5, 'DisplayName', ScaleRows{k});
end
PlotReference(RefMean, RefStd);
set(gca, 'XScale', 'log'); xticks([0.1 1 10]); grid on
xlabel('Parameter (multiple of data scale)'); ylabel('Weighted MSE ((% body fat)^2)');
title('(a) Scale-type kernels'); legend('Location', 'northeast');

subplot(1,2,2); hold on
for k = 1:numel(ExpRows)
    r = strcmp(Rows(:,2), ExpRows{k});
    plot(ExpVals{k}, Err(r,:), '-o', 'LineWidth', 1.5, 'DisplayName', ExpRows{k});
end
PlotReference(RefMean, RefStd);
grid on; xlabel('Exponent / degree'); ylabel('Weighted MSE ((% body fat)^2)');
title('(b) Exponent-type kernels'); legend('Location', 'northwest');
sgtitle('BodyFat: effect of kernel parameter on KPCA error (d = 3)');
saveas(gcf, 'Fig_KernelParameterEffect.png');

%% Figure B: Gaussian KPCA embedding at small/medium/large sigma (no ANN needed)
figure(21); clf
Mult = [0.1 1 10];
for i = 1:3
    Kt.Type = 'Gaussian';  Kt.Parameter = Mult(i) * Dmed;
    Z = KernelPCA_Incomplete(Data.Input', Kt, 2);
    subplot(1,3,i)
    scatter(Z(1,:), Z(2,:), 20, Data.Output, 'filled'); grid on
    xlabel('Component 1'); ylabel('Component 2');
    title(sprintf('\\sigma = %.1f \\times median distance', Mult(i)));
end
cb = colorbar; cb.Label.String = 'Body fat (%)';
sgtitle('Gaussian KPCA embedding vs. bandwidth (BodyFat)');
saveas(gcf, 'Fig_GaussianEmbeddings.png');

%% Figure C: every kernel at its best parameter, re-evaluated over 5 fresh seeds
Seeds = 21:25;  NumExec = 50;                  % use NumExec = 2 to debug
nR = size(Rows,1);
BestP = zeros(nR,1);  BestMean = zeros(nR,1);  BestStd = zeros(nR,1);
for r = 1:nR
    [~, j]   = min(Err(r,:));                  % best of the three tested values
    BestP(r) = Rows{r,3}(j);
    e = arrayfun(@(s) EvalKPCA(Data, Rows{r,2}, BestP(r), d, WeightTrain, NumExec, s), Seeds);
    BestMean(r) = mean(e);  BestStd(r) = std(e);
    fprintf('%-18s best param %-10.4g -> %.2f +/- %.2f\n', Rows{r,2}, BestP(r), BestMean(r), BestStd(r));
end

[~, order] = sort(BestMean);
figure(22); clf
bar(BestMean(order)); hold on
errorbar(1:nR, BestMean(order), BestStd(order), 'k.', 'LineWidth', 1.2);
yline(RefMean, '--k', 'Linear / PCA reference');
xticks(1:nR); xticklabels(Rows(order,2)); xtickangle(30); grid on
ylabel('Weighted MSE ((% body fat)^2)');
title('KPCA kernels at their best tested parameter (d = 3, mean \pm std over 5 seeds)');
saveas(gcf, 'Fig_KernelComparison.png');

BestTable = table(Rows(:,2), BestP, BestMean, BestStd, ...
    'VariableNames', {'Kernel','BestParam','MeanError','StdError'});
save('KernelFigures_Results.mat', 'BestTable');

%% Local functions (must stay at the end of the file)
function PlotReference(m, s)
    yline(m, '--k', 'Linear / PCA', 'HandleVisibility', 'off');
    yline(m + 2*s, ':k', 'HandleVisibility', 'off');      % +/- 2 std noise band
    yline(m - 2*s, ':k', 'HandleVisibility', 'off');
end

function err = EvalKPCA(Data, Type, Param, d, WeightTrain, nExec, seed)
    Kt.Type = Type;  Kt.Parameter = Param;
    rng(seed);
    try
        Z = KernelPCA_Incomplete(Data.Input', Kt, d);
        if ~isreal(Z) || any(~isfinite(Z(:))), error('bad embedding'); end
        Dt = Data;  Dt.Input = Z;  Dt.d = d;
        err = FeatureSelectionObjective(ones(1,d), Dt, WeightTrain, nExec);
    catch
        err = NaN;
    end
end