%% astar_routes.m
% MECE 694 Final Project - Ski-touring route optimization, Rogers Pass
% Step 4: Single-objective route finding with A*.
%
% For each destination, find:
%   - the FASTEST route        (minimize total time)
%   - the LEAST-EFFORT route   (minimize total energy)
% then compare them. If the two routes are nearly identical, time and effort
% are redundant objectives; if they diverge, both belong in the multi-objective
% optimization.
%
% Also:
%   - Dijkstra (A* with heuristic = 0) to show how much work the heuristic saves
%   - A check against MATLAB's built-in shortestpath, to verify our A* is correct
%
% Inputs:  data/terrain.mat, data/features.mat, data/costs.mat
% Output:  data/routes.mat

clear; clc; close all;
dataDir = 'data';
load(fullfile(dataDir, 'terrain.mat'),  'dtm');
load(fullfile(dataDir, 'features.mat'), 'hillshade');
load(fullfile(dataDir, 'costs.mat'),    'timeH', 'effortJ', 'noGo', 'p', 'dirs', 'edgeLen', 'R', 'cs');
[nR, nC, nD] = size(timeH);
N = nR * nC;

runDijkstraCompare = true;   % run Dijkstra too (slower) to compare nodes expanded
runMatlabCheck     = true;   % verify against MATLAB's digraph/shortestpath

%% 1. Trips: start and destinations (lat, lon in WGS84)
startName = 'Connaught parking';
startLL   = [51.3024361, -117.5220423];

dest = struct( ...
    'name', {'Balu Pass', 'Video Peak', '8812 Bowl', 'Ursus Trees', 'Little Sifton'}, ...
    'll',   {[51.2853801, -117.5817972], [51.3025998, -117.5856377], ...
             [51.2969848, -117.5905893], [51.3036670, -117.5721980], ...
             [51.3275145, -117.5528187]});

%% 2. Convert lat/lon to grid cells (snapping to the nearest passable cell if needed)
[sr, sc] = latlon2cell(startLL, R);
[sr, sc] = snapToPassable(sr, sc, noGo, startName);
startIdx = sub2ind([nR nC], sr, sc);

for j = 1:numel(dest)
    [r, c] = latlon2cell(dest(j).ll, R);
    [r, c] = snapToPassable(r, c, noGo, dest(j).name);
    dest(j).rc  = [r c];
    dest(j).idx = sub2ind([nR nC], r, c);
end

%% 3. Heuristic constants (must UNDERESTIMATE the true remaining cost = admissible)
% Time:   any route must cover at least the straight-line horizontal distance d
%         and climb at least the net elevation gain dz, so the Munter time for
%         (d, dz) is a lower bound.
% Effort: two lower bounds, take the larger:
%         (a) cheapest possible energy per metre (Minetti minimum) x distance d
%         (b) cheapest possible energy per metre CLIMBED x net gain dz
iAll   = linspace(-0.45, 0.45, 4001);
CwMin  = min(minettiCw(iAll));                                   % J/kg/m
iUp    = linspace(1e-3, tand(p.maxSkinAngle), 4000);
eUpMin = min(minettiCw(iUp) .* sqrt(1 + iUp.^2) ./ iUp);        % J/kg per m climbed
fprintf('Heuristic constants: min cost %.2f J/kg/m, min climb cost %.1f J/kg per m gained\n', ...
        CwMin, eUpMin);

[Cc, Rr] = meshgrid(1:nC, 1:nR);    % column / row index of every cell

%% 4. Optional: build MATLAB digraphs once (for the correctness check)
objNames  = {'time', 'effort'};
costCubes = {timeH, effortJ};
if runMatlabCheck
    fprintf('Building MATLAB digraphs for verification...\n');
    G = cell(1, 2);
    for o = 1:2
        G{o} = buildDigraph(costCubes{o}, dirs);
    end
end

%% 5. Run A* for every destination and objective
res = struct();
fprintf('\n%-14s %-7s %8s %9s %8s %7s %10s %10s %8s\n', 'Destination', 'Minimize', ...
        'Time', 'Effort', 'Length', 'Gain', 'A* nodes', 'Dijk nodes', 'A* time');
fprintf('%s\n', repmat('-', 1, 92));

for j = 1:numel(dest)
    goal = dest(j).idx;
    [rg, cg] = ind2sub([nR nC], goal);

    % Heuristic maps for this goal
    d    = cs * hypot(Rr - rg, Cc - cg);         % straight-line horizontal distance (m)
    dzUp = max(dtm(goal) - dtm, 0);              % net climb still needed (m)
    h.time   = (d/1000/p.kmPerUnit + dzUp/p.vertPerUnit) / p.unitsPerHour;
    h.effort = max(CwMin * d, eUpMin * dzUp);

    for o = 1:2
        obj = objNames{o};

        tic;
        [path, cost, nExp] = astarGrid(costCubes{o}, startIdx, goal, h.(obj)(:), dirs);
        tA = toc;
        if isempty(path)
            warning('No route found to %s (%s).', dest(j).name, obj);
            continue
        end

        nDij = NaN;
        if runDijkstraCompare
            [~, costD, nDij] = astarGrid(costCubes{o}, startIdx, goal, zeros(N, 1), dirs);
            assert(abs(costD - cost) <= 1e-9 * max(1, cost), 'A* and Dijkstra disagree!');
        end

        if runMatlabCheck
            [~, costM] = shortestpath(G{o}, startIdx, goal);
            assert(abs(costM - cost) <= 1e-9 * max(1, cost), ...
                   'Our A* disagrees with MATLAB shortestpath (%s, %s)', dest(j).name, obj);
        end

        s = pathStats(path, timeH, effortJ, dtm, dirs, edgeLen, p);
        res(j, o).path  = path;
        res(j, o).stats = s;
        res(j, o).nExp  = nExp;
        res(j, o).nDij  = nDij;

        fprintf('%-14s %-7s %6.0f m %6.0f kc %5.2f km %5.0f m %10d %10d %6.2f s\n', ...
                dest(j).name, obj, s.timeMin, s.effortKcal, s.lengthKm, s.gainM, ...
                nExp, nDij, tA);
    end
end
if runMatlabCheck
    fprintf('\nAll routes match MATLAB shortestpath -> our A* is returning optimal paths.\n');
end

%% 6. Route-level comparison: do time and effort pick different routes?
fprintf('\n--- Fastest vs least-effort route ---\n');
fprintf('%-14s %10s %10s %14s %16s\n', 'Destination', 'Time cost', 'Effort gain', ...
        'Mean sep. (m)', '% within 60 m');
for j = 1:numel(dest)
    if isempty(res(j,1).path) || isempty(res(j,2).path), continue; end
    sT = res(j,1).stats;  sE = res(j,2).stats;
    % "Time cost": how much SLOWER the least-effort route is (%)
    % "Effort gain": how much LESS effort it uses (%)
    dTime   = 100 * (sE.timeMin - sT.timeMin) / sT.timeMin;
    dEffort = 100 * (sT.effortKcal - sE.effortKcal) / sT.effortKcal;
    [meanSep, pctClose] = routeSeparation(res(j,1).path, res(j,2).path, nR, cs);
    fprintf('%-14s %9.1f%% %10.1f%% %14.0f %15.0f%%\n', ...
            dest(j).name, dTime, dEffort, meanSep, pctClose);
end
fprintf('(Small %% differences and close routes => effort is largely redundant with time.)\n');

%% 7. Map of the routes
figure('Name', 'A* routes', 'Position', [80 80 1100 850]);
xC = R.XWorldLimits + [0.5 -0.5]*cs;
yC = fliplr(R.YWorldLimits + [0.5 -0.5]*cs);
imagesc(xC, yC, hillshade); colormap(gray); clim([0 1]);
set(gca, 'YDir', 'normal'); axis image; hold on;

% No-go cells in translucent red
red = cat(3, ones(nR, nC), zeros(nR, nC), zeros(nR, nC));
image(xC, yC, red, 'AlphaData', 0.30 * noGo);

cols = lines(numel(dest));
hLeg = gobjects(0);  legTxt = {};
allX = [];  allY = [];
for j = 1:numel(dest)
    for o = 1:2
        if isempty(res(j,o).path), continue; end
        [x, y] = cellXY(res(j,o).path, nR, R, cs);
        allX = [allX; x];  allY = [allY; y]; %#ok<AGROW>
        style = '-';  if o == 2, style = '--'; end
        hl = plot(x, y, style, 'Color', cols(j,:), 'LineWidth', 2);
        if o == 1, hLeg(end+1) = hl; legTxt{end+1} = dest(j).name; end %#ok<SAGROW>
    end
    [gx, gy] = cellXY(dest(j).idx, nR, R, cs);
    plot(gx, gy, '^', 'MarkerFaceColor', cols(j,:), 'MarkerEdgeColor', 'k', 'MarkerSize', 9);
end
[sx, sy] = cellXY(startIdx, nR, R, cs);
hS = plot(sx, sy, 's', 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'MarkerSize', 11);
hold off;

pad = 600;
xlim([min(allX)-pad, max(allX)+pad]);  ylim([min(allY)-pad, max(allY)+pad]);
legend([hLeg hS], [legTxt {'Start'}], 'Location', 'bestoutside');
title({'A* routes from Connaught parking', ...
       'solid = fastest, dashed = least effort, red = no-go (>38\circ)'});
xlabel('Easting (m)'); ylabel('Northing (m)');

%% 8. Save
save(fullfile(dataDir, 'routes.mat'), 'res', 'dest', 'startIdx', 'startName', 'objNames');
fprintf('Saved routes.mat\n');

%% ======================= Local functions =======================

function [path, cost, nExpanded] = astarGrid(costCube, startIdx, goalIdx, hVec, dirs)
% A* on an 8-connected grid with DIRECTED edge costs.
%   costCube(r,c,k) = cost of moving from cell (r,c) in direction k (Inf = blocked)
%   hVec(n)         = heuristic (lower bound on remaining cost) for node n
% With hVec = 0 this is exactly Dijkstra's algorithm.
% Uses a binary min-heap written inline (MATLAB has no built-in priority queue,
% and calling a helper function per push/pop would copy the arrays each time).
    [nR, nC, nD] = size(costCube);
    N       = nR * nC;
    costLin = reshape(costCube, N, nD);
    offs    = dirs(:,1) + dirs(:,2) * nR;      % linear-index offset of each direction

    g      = inf(N, 1);  g(startIdx) = 0;      % best known cost from start
    parent = zeros(N, 1);
    closed = false(N, 1);

    cap = 8*N + 16;                            % heap storage (lazy deletion -> duplicates)
    hk  = zeros(cap, 1);  hn = zeros(cap, 1);  hs = 1;
    hk(1) = hVec(startIdx);  hn(1) = startIdx;
    nExpanded = 0;

    while hs > 0
        % ---- pop the node with the smallest f = g + h ----
        u = hn(1);
        hk(1) = hk(hs);  hn(1) = hn(hs);  hs = hs - 1;
        i = 1;
        while true                             % sift down
            l = 2*i;
            if l > hs, break; end
            m = l;
            if l + 1 <= hs && hk(l+1) < hk(l), m = l + 1; end
            if hk(m) < hk(i)
                tk = hk(i); hk(i) = hk(m); hk(m) = tk;
                tn = hn(i); hn(i) = hn(m); hn(m) = tn;
                i = m;
            else
                break
            end
        end

        if closed(u), continue; end            % stale duplicate entry
        closed(u) = true;
        nExpanded = nExpanded + 1;
        if u == goalIdx, break; end

        % ---- relax the 8 outgoing moves ----
        ru = mod(u - 1, nR) + 1;
        cu = floor((u - 1) / nR) + 1;
        for k = 1:nD
            rv = ru + dirs(k,1);  cv = cu + dirs(k,2);
            if rv < 1 || rv > nR || cv < 1 || cv > nC, continue; end
            v = u + offs(k);
            if closed(v), continue; end
            w = costLin(u, k);
            if ~isfinite(w), continue; end
            gv = g(u) + w;
            if gv < g(v)
                g(v) = gv;  parent(v) = u;
                % ---- push v with key f = g + h ----
                hs = hs + 1;  hk(hs) = gv + hVec(v);  hn(hs) = v;
                i = hs;
                while i > 1                    % sift up
                    pI = floor(i/2);
                    if hk(pI) > hk(i)
                        tk = hk(i); hk(i) = hk(pI); hk(pI) = tk;
                        tn = hn(i); hn(i) = hn(pI); hn(pI) = tn;
                        i = pI;
                    else
                        break
                    end
                end
            end
        end
    end

    cost = g(goalIdx);
    if ~isfinite(cost), path = []; return; end
    path = goalIdx;                            % walk back through the parents
    while path(1) ~= startIdx
        path = [parent(path(1)); path]; %#ok<AGROW>
    end
end

function G = buildDigraph(costCube, dirs)
% Same graph as astarGrid uses, built as a MATLAB digraph (for verification).
    [nR, nC, nD] = size(costCube);
    [Cc, Rr] = meshgrid(1:nC, 1:nR);
    s = cell(nD,1); t = cell(nD,1); w = cell(nD,1);
    for k = 1:nD
        rv = Rr + dirs(k,1);  cv = Cc + dirs(k,2);
        ck = costCube(:,:,k);
        ok = rv >= 1 & rv <= nR & cv >= 1 & cv <= nC & isfinite(ck);
        s{k} = find(ok);
        t{k} = sub2ind([nR nC], rv(ok), cv(ok));
        w{k} = ck(ok);
    end
    G = digraph(vertcat(s{:}), vertcat(t{:}), vertcat(w{:}), nR*nC);
end

function s = pathStats(path, timeH, effortJ, dtm, dirs, edgeLen, p)
% Evaluate BOTH objectives (and descriptive stats) along a route.
    [nR, nC, nD] = size(timeH);
    [r, c] = ind2sub([nR nC], path);
    kTab = zeros(3);                           % (d_row, d_col) -> direction index
    for k = 1:nD, kTab(dirs(k,1)+2, dirs(k,2)+2) = k; end
    kk  = kTab(sub2ind([3 3], diff(r)+2, diff(c)+2));
    idx = sub2ind([nR nC nD], r(1:end-1), c(1:end-1), kk);
    z   = dtm(path);
    s.timeMin    = 60 * sum(timeH(idx));
    s.effortKcal = sum(effortJ(idx)) * p.mass / 4184;
    s.lengthKm   = sum(edgeLen(kk)) / 1000;
    s.gainM      = sum(max(diff(z), 0));
    s.maxGrade   = max(atand(diff(z) ./ edgeLen(kk)));
end

function [meanSep, pctClose] = routeSeparation(pathA, pathB, nR, cs)
% How far apart are two routes? Mean nearest-cell distance (both directions)
% and % of route B within 60 m (2 cells) of route A - a corridor-scale measure.
    [rA, cA] = ind2sub([nR 1e6], pathA);
    [rB, cB] = ind2sub([nR 1e6], pathB);
    D = cs * hypot(rA - rB', cA - cB');        % |A| x |B| distance matrix (m)
    dAB = min(D, [], 2);  dBA = min(D, [], 1)';
    meanSep  = mean([dAB; dBA]);
    pctClose = 100 * mean(dBA <= 60);
end

function [r, c] = latlon2cell(ll, R)
    [x, y] = projfwd(R.ProjectedCRS, ll(1), ll(2));
    [r, c] = worldToDiscrete(R, x, y);
    if isnan(r), error('Point (%.5f, %.5f) is outside the grid.', ll(1), ll(2)); end
end

function [r, c] = snapToPassable(r, c, noGo, name)
% If a point lands on a no-go cell (e.g. a steep summit), move it to the
% nearest passable cell and say so.
    if ~noGo(r, c), return; end
    [nR, nC] = size(noGo);
    for rad = 1:10
        [dc, dr] = meshgrid(-rad:rad, -rad:rad);
        rr = r + dr(:);  cc = c + dc(:);
        ok = rr >= 1 & rr <= nR & cc >= 1 & cc <= nC;
        rr = rr(ok);  cc = cc(ok);
        pass = ~noGo(sub2ind([nR nC], rr, cc));
        if any(pass)
            rr = rr(pass);  cc = cc(pass);
            [~, m] = min(hypot(rr - r, cc - c));
            fprintf('Note: %s was on a no-go cell; moved %.0f m to nearest passable cell.\n', ...
                    name, 30 * hypot(rr(m) - r, cc(m) - c));
            r = rr(m);  c = cc(m);
            return
        end
    end
    error('%s: no passable cell within 300 m.', name);
end

function [x, y] = cellXY(idx, nR, R, cs)
% Grid cell -> UTM coordinates of the cell centre (row 1 = north)
    [r, c] = ind2sub([nR 1e6], idx);
    x = R.XWorldLimits(1) + (c - 0.5) * cs;
    y = R.YWorldLimits(2) - (r - 0.5) * cs;
end

function C = minettiCw(i)
% Minetti et al. (2002) energy cost of walking, J/kg/m (same as time_effort_models.m)
    C = 280.5*i.^5 - 58.7*i.^4 - 76.8*i.^3 + 51.9*i.^2 + 19.6*i + 2.5;
end
