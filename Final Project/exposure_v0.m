%% exposure_v0.m
% MECE 694 Final Project - Ski-touring route optimization, Rogers Pass
% Step 5: Placeholder avalanche EXPOSURE model + weighted-sum A* trade-off.
%
% "v0" = terrain-only, slope-based, no forest data and no danger rating yet.
% Its job is to test the whole pipeline end to end and produce a first
% time-vs-exposure trade-off. It will be replaced by the fuzzy / classifier
% exposure model (Stage 2) - the rest of the pipeline stays the same.
%
% Exposure of a cell = max( release-area potential , overhead exposure )
%   - release-area potential (PRA): is this slope steep enough to avalanche?
%   - overhead exposure: could an avalanche starting ABOVE reach this cell?
% Taking the max of the two mirrors how AutoATES combines its slope and
% runout layers (each cell gets the worst of the ratings).
%
% Inputs:  data/terrain.mat, features.mat, costs.mat, routes.mat
% Output:  data/exposure_v0.mat

clear; clc; close all;
dataDir = 'data';
load(fullfile(dataDir, 'terrain.mat'),  'dtm', 'canopy');
load(fullfile(dataDir, 'features.mat'), 'slope', 'hillshade');
load(fullfile(dataDir, 'costs.mat'),    'timeH', 'noGo', 'p', 'dirs', 'edgeLen', 'R', 'cs');
load(fullfile(dataDir, 'routes.mat'),   'dest', 'startIdx');
[nR, nC, nD] = size(timeH);
N = nR * nC;

%% 1. Parameters (all placeholders - v1 replaces this whole model)
% Release-area potential: trapezoid on slope angle. Most slab avalanches
% release on 30-45 deg slopes; very steep rock sheds snow before it builds up.
e.praLow   = 28;   % deg: PRA starts rising
e.praFull  = 35;   % deg: PRA = 1
e.praHigh  = 45;   % deg: PRA starts falling
e.praZero  = 55;   % deg: PRA = 0 again
e.praMin   = 0.5;  % cells with PRA >= this act as avalanche sources for runout

% Runout: alpha-angle method. An avalanche keeps running while the angle of
% the line from its starting point to its current position is >= alpha.
% 24 deg is the Flow-Py setting used by Sykes et al. (2024) at Rogers Pass.
e.alpha    = 24;   % deg
e.spread   = 1;    % cells of lateral spreading each side of the flow line (+/-30 m)

% Optional forest effect using the (weak) DSM-DTM canopy proxy. Off by default;
% v1 will use BC VRI forest data instead.
e.useCanopy       = false;
e.canopyForest    = 10;    % m: canopy taller than this counts as forest
e.forestReduction = 0.7;   % forest reduces release potential by 70%

% Weighted-sum sweep: cost = time + w * exposure (both in minutes).
% w = how many minutes of extra travel you'd accept to avoid 1 exposed minute.
wList = [0 0.25 0.5 1 2 4 8 16 32 64];

%% 2. Release-area potential (PRA), 0..1
pra = zeros(nR, nC);
m = slope >= e.praLow & slope < e.praFull;
pra(m) = (slope(m) - e.praLow) / (e.praFull - e.praLow);
pra(slope >= e.praFull & slope <= e.praHigh) = 1;
m = slope > e.praHigh & slope < e.praZero;
pra(m) = (e.praZero - slope(m)) / (e.praZero - e.praHigh);
if e.useCanopy
    f = canopy > e.canopyForest;
    pra(f) = pra(f) * (1 - e.forestReduction);
end

%% 3. Flow direction: D8 steepest descent
% Each cell sends snow to the neighbour with the steepest downhill drop.
bestDrop = zeros(nR, nC);  bestDir = zeros(nR, nC);
for k = 1:nD
    dr = dirs(k,1);  dc = dirs(k,2);
    zk = NaN(nR, nC);
    rows = max(1, 1-dr) : min(nR, nR-dr);
    cols = max(1, 1-dc) : min(nC, nC-dc);
    zk(rows, cols) = dtm(rows+dr, cols+dc);
    dropk = (dtm - zk) / edgeLen(k);           % positive = downhill
    better = dropk > bestDrop;                 % NaN comparisons are false
    bestDrop(better) = dropk(better);
    bestDir(better)  = k;
end
offs = dirs(:,1) + dirs(:,2) * nR;
rec  = (1:N)';                                 % receiver; pits point to themselves
hasRec = bestDir(:) > 0;
rec(hasRec) = rec(hasRec) + offs(bestDir(hasRec));

%% 4. Runout: trace every source cell downhill (vectorised over all sources)
src = find(pra(:) >= e.praMin);
pos = src;  z0 = dtm(src);  L = zeros(size(src));
counts = zeros(N, 1);                          % how many sources reach each cell
for it = 1:2000
    nxt   = rec(pos);
    moved = nxt ~= pos;
    kk = bestDir(pos);
    L(moved) = L(moved) + edgeLen(kk(moved));
    alphaNow = atand((z0 - dtm(nxt)) ./ max(L, eps));
    active = moved & alphaNow >= e.alpha;
    pos = nxt(active);  z0 = z0(active);  L = L(active);
    if isempty(pos), break; end
    counts = counts + accumarray(pos, 1, [N 1]);
end
counts = reshape(counts, nR, nC);
counts = maxFilter(counts, e.spread);          % widen the single-cell flow lines

% Log-scale and normalise to 0..1 (one big path vs many small ones)
overhead = log1p(counts) / log1p(max(counts(:)));

%% 5. Cell exposure and per-move exposure cost
exposure = max(pra, overhead);

% Exposure cost of a move = exposure x time spent making it (exposed MINUTES).
% Risk grows with the time spent in avalanche terrain, so slow zigzagging
% through a start zone counts for more than skiing quickly across it.
% Half the move is spent in each of the two cells, so use their average.
timeMin = 60 * timeH;
expoMin = NaN(nR, nC, nD);
for k = 1:nD
    dr = dirs(k,1);  dc = dirs(k,2);
    Et = NaN(nR, nC);
    rows = max(1, 1-dr) : min(nR, nR-dr);
    cols = max(1, 1-dc) : min(nC, nC-dc);
    Et(rows, cols) = exposure(rows+dr, cols+dc);
    ek = 0.5 * (exposure + Et) .* timeMin(:,:,k);
    ek(isinf(timeMin(:,:,k))) = Inf;           % keep blocked moves blocked
    expoMin(:,:,k) = ek;
end

fprintf('--- Exposure v0 ---\n');
fprintf('Release sources: %d cells (%.1f%% of area)\n', numel(src), 100*numel(src)/N);
fprintf('Cells reached by at least one avalanche: %.1f%%\n', 100*mean(counts(:) > 0));
fprintf('Cells with exposure > 0.5: %.1f%%\n', 100*mean(exposure(:) > 0.5));

%% 6. Weighted-sum A*: sweep w for every destination
[Cc, Rr] = meshgrid(1:nC, 1:nR);
nW = numel(wList);
front = struct();

fprintf('\n%-14s %6s %9s %11s %12s %13s\n', 'Destination', 'w', 'Time', 'Exposure', ...
        'Time +%', 'Exposure -%');
fprintf('%s\n', repmat('-', 1, 70));
for j = 1:numel(dest)
    goal = dest(j).idx;
    [rg, cg] = ind2sub([nR nC], goal);
    d    = cs * hypot(Rr - rg, Cc - cg);
    dzUp = max(dtm(goal) - dtm, 0);
    % Time heuristic in minutes. Still admissible for time + w*exposure,
    % because the exposure term can never be negative.
    hT = 60 * (d/1000/p.kmPerUnit + dzUp/p.vertPerUnit) / p.unitsPerHour;

    T = NaN(nW, 1);  X = NaN(nW, 1);  paths = cell(nW, 1);
    for iw = 1:nW
        cube = timeMin + wList(iw) * expoMin;
        path = astarGrid(cube, startIdx, goal, hT(:), dirs);
        paths{iw} = path;
        T(iw) = pathCost(path, timeMin, dirs, nR, nC);
        X(iw) = pathCost(path, expoMin, dirs, nR, nC);
        fprintf('%-14s %6.2f %7.0f m %9.1f m %11.1f%% %12.1f%%\n', dest(j).name, ...
                wList(iw), T(iw), X(iw), 100*(T(iw)/T(1) - 1), 100*(1 - X(iw)/X(1)));
    end
    nUnique = size(unique(round([T X], 6), 'rows'), 1);
    fprintf('%-14s -> %d distinct routes from %d weights\n\n', '', nUnique, nW);
    front(j).T = T;  front(j).X = X;  front(j).paths = paths;
end

%% 7. Figures
xC = R.XWorldLimits + [0.5 -0.5]*cs;
yC = fliplr(R.YWorldLimits + [0.5 -0.5]*cs);

% (a) Exposure map over hillshade
figure('Name', 'Exposure v0', 'Position', [60 60 1300 520]);
tiledlayout(1, 3, 'TileSpacing', 'compact');
layers = {pra, overhead, exposure};
names  = {'Release-area potential', 'Overhead exposure (runout)', 'Exposure = max of both'};
for t = 1:3
    nexttile;
    imagesc(xC, yC, hillshade); colormap(gca, gray); clim([0 1]);
    set(gca, 'YDir', 'normal'); axis image; hold on;
    hotLayer = layers{t};
    rgb = cat(3, ones(nR,nC), 0.4*(1 - hotLayer), zeros(nR,nC));   % yellow -> red
    image(xC, yC, rgb, 'AlphaData', 0.75 * hotLayer);
    hold off; title(names{t}); xlabel('Easting (m)');
end

% (b) Routes for each w, one panel per destination
figure('Name', 'Weighted-sum routes', 'Position', [40 40 1500 850]);
tiledlayout(2, 3, 'TileSpacing', 'compact');
cmapW = parula(nW);
for j = 1:numel(dest)
    nexttile;
    imagesc(xC, yC, hillshade); colormap(gca, gray); clim([0 1]);
    set(gca, 'YDir', 'normal'); axis image; hold on;
    rgb = cat(3, ones(nR,nC), 0.4*(1 - exposure), zeros(nR,nC));
    image(xC, yC, rgb, 'AlphaData', 0.45 * exposure);
    allX = []; allY = [];
    for iw = 1:nW
        [x, y] = cellXY(front(j).paths{iw}, nR, R, cs);
        plot(x, y, '-', 'Color', cmapW(iw,:), 'LineWidth', 1.8);
        allX = [allX; x]; allY = [allY; y]; %#ok<AGROW>
    end
    [sx, sy] = cellXY(startIdx, nR, R, cs);
    [gx, gy] = cellXY(dest(j).idx, nR, R, cs);
    plot(sx, sy, 's', 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'MarkerSize', 9);
    plot(gx, gy, '^', 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', 'MarkerSize', 9);
    hold off;
    pad = 400;
    xlim([min(allX)-pad max(allX)+pad]); ylim([min(allY)-pad max(allY)+pad]);
    title(dest(j).name);
end
nexttile; axis off;
colormap(gca, cmapW); clim([0.5 nW+0.5]);
cb = colorbar('Location', 'west'); cb.Ticks = 1:nW;
cb.TickLabels = arrayfun(@(w) sprintf('w = %g', w), wList, 'UniformOutput', false);
title('Route colour = exposure weight');

% (c) Trade-off (Pareto) plots
figure('Name', 'Time vs exposure trade-off', 'Position', [100 100 1100 650]);
tiledlayout(2, 3, 'TileSpacing', 'compact');
for j = 1:numel(dest)
    nexttile;
    plot(front(j).T, front(j).X, '-', 'Color', [0.6 0.6 0.6]); hold on;
    scatter(front(j).T, front(j).X, 50, 1:nW, 'filled', 'MarkerEdgeColor', 'k');
    colormap(gca, cmapW); hold off; grid on;
    xlabel('Time (min)'); ylabel('Exposure (exposed min)');
    title(dest(j).name);
end

%% 8. Save
save(fullfile(dataDir, 'exposure_v0.mat'), 'pra', 'overhead', 'exposure', 'counts', ...
     'expoMin', 'timeMin', 'front', 'wList', 'e');
fprintf('Saved exposure_v0.mat\n');

%% ======================= Local functions =======================
function c = pathCost(path, cube, dirs, nR, nC)
% Sum a per-move cost cube along a path.
    [r, cl] = ind2sub([nR nC], path);
    kTab = zeros(3);
    for k = 1:size(dirs,1), kTab(dirs(k,1)+2, dirs(k,2)+2) = k; end
    kk  = kTab(sub2ind([3 3], diff(r)+2, diff(cl)+2));
    idx = sub2ind([nR nC size(cube,3)], r(1:end-1), cl(1:end-1), kk);
    c = sum(cube(idx));
end

function B = maxFilter(A, rad)
% Moving maximum over a (2*rad+1)^2 window (no Image Processing Toolbox needed).
    if rad == 0, B = A; return; end
    [nR, nC] = size(A);
    P = -inf(nR + 2*rad, nC + 2*rad);
    P(rad+1:rad+nR, rad+1:rad+nC) = A;
    B = -inf(nR, nC);
    for dr = 0:2*rad
        for dc = 0:2*rad
            B = max(B, P(1+dr:nR+dr, 1+dc:nC+dc));
        end
    end
end

function [x, y] = cellXY(idx, nR, R, cs)
% Grid cell -> UTM coordinates of the cell centre (row 1 = north)
    [r, c] = ind2sub([nR 1e6], idx);
    x = R.XWorldLimits(1) + (c - 0.5) * cs;
    y = R.YWorldLimits(2) - (r - 0.5) * cs;
end
