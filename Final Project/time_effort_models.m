%% time_effort_models.m
% MECE 694 Final Project - Ski-touring route optimization, Rogers Pass
% Step 3: Time and effort cost models for the ASCENT (skinning).
%
% Part A - "Climb H metres at angle theta" analysis:
%          Do time and effort have different optimal climbing angles?
%          If yes, they genuinely conflict and both belong in the optimization.
% Part B - Build per-edge time and effort costs for the whole graph
%          (480 x 570 x 8), apply the no-go mask, and save them.
%
% Input:  data/features.mat  (from terrain_features.m)
% Output: data/costs.mat

clear; clc; close all;
dataDir = 'data';
load(fullfile(dataDir, 'features.mat'), 'slope', 'edgeGrade', 'dirs', 'edgeLen', 'R', 'cs');
[nR, nC, nDir] = size(edgeGrade);

%% 1. Model parameters (assumptions - calibrate these, e.g. from your GPS tracks)
% --- Time: Munter method (standard Swiss/European ski-touring planning rule)
% 1 "unit" = 1 km horizontal OR 100 m of ascent; skinning uphill covers ~4 units/hour.
p.unitsPerHour  = 4;
p.kmPerUnit     = 1;      % km horizontal per unit
p.vertPerUnit   = 100;    % m ascent per unit

% --- Skin track geometry
p.maxSkinAngle  = 20;     % deg - steepest comfortable skin track grade.
                          % Steeper climbs are done by zigzagging at this angle.
p.kickTurnEvery = 15;     % m of vertical gained per kick turn when zigzagging
p.kickTurnTime  = 20;     % s per kick turn

% --- Terrain limits
p.maxCellSlope  = 38;     % deg - cells steeper than this are impassable for skinning.
                          % Set below the real-world limit because a 30 m DEM
                          % underestimates slope (smoothing).

% --- Effort: converting J/kg to something readable
p.mass          = 85;     % kg, skier + gear

%% ===================== PART A: climbing-angle analysis =====================
% Climb a fixed height H at a constant grade theta. Both time and effort scale
% linearly with H, so the SHAPE of the curves (and the optimal angles) does not
% depend on H - we just pick 500 m for readable numbers.
H     = 500;                         % m
theta = (2:0.25:35)';                % candidate climbing grades (deg)
LhA   = H ./ tand(theta);            % horizontal distance needed at each grade
dzA   = H * ones(size(theta));

T = timeCost(LhA, dzA, p) * 60;                  % minutes
E = effortCost(LhA, dzA, p) * p.mass / 4184;     % kcal (J/kg * kg / 4184 J/kcal)

[Tmin, iT] = min(T);   thetaT = theta(iT);
[Emin, iE] = min(E);   thetaE = theta(iE);

fprintf('--- Part A: climbing %d m ---\n', H);
fprintf('Fastest grade:      %.1f deg  (%.0f min)\n', thetaT, Tmin);
fprintf('Least-effort grade: %.1f deg  (%.0f kcal)\n', thetaE, Emin);
if abs(thetaT - thetaE) > 1
    fprintf('-> Optimal angles differ: time and effort CONFLICT between %.1f and %.1f deg.\n', ...
            min(thetaT, thetaE), max(thetaT, thetaE));
else
    fprintf('-> Optimal angles nearly coincide: time and effort barely conflict.\n');
end
fprintf('Implied vertical rate at %d deg: %.0f m/h  (check against your GPS tracks)\n', ...
        p.maxSkinAngle, H / (timeCost(H/tand(p.maxSkinAngle), H, p)));

figure('Name', 'Climbing angle trade-off');
plot(theta, T/Tmin, 'b-', 'LineWidth', 2); hold on;
plot(theta, E/Emin, 'r-', 'LineWidth', 2);
plot(thetaT, 1, 'bo', 'MarkerFaceColor', 'b');
plot(thetaE, 1, 'ro', 'MarkerFaceColor', 'r');
yl = ylim;
patch([min(thetaT,thetaE) max(thetaT,thetaE) max(thetaT,thetaE) min(thetaT,thetaE)], ...
      [yl(1) yl(1) yl(2) yl(2)], [0.5 0.5 0.5], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
xline(p.maxSkinAngle, '--k', 'max skin angle');
hold off; grid on; ylim(yl);
xlabel('Climbing grade \theta (deg)');
ylabel('Cost relative to its own minimum');
legend('Time (Munter + zigzag)', 'Effort (Minetti energy cost)', ...
       'fastest', 'least effort', 'conflict zone', 'Location', 'north');
title(sprintf('Climbing %d m: where do time and effort disagree?', H));

%% ===================== PART B: per-edge costs for the graph =====================
% Horizontal length of each of the 8 moves, as a 1x1x8 array
Lh3 = repmat(reshape(edgeLen, 1, 1, nDir), nR, nC, 1);
dz  = tand(edgeGrade) .* Lh3;                    % elevation change of each move (m)

timeH   = timeCost(Lh3, dz, p);                  % hours per move
effortJ = effortCost(Lh3, dz, p);                % J/kg per move

% No-go mask: a move is blocked if it ENTERS a cell steeper than maxCellSlope.
noGo = slope > p.maxCellSlope;
for k = 1:nDir
    dr = dirs(k, 1);  dc = dirs(k, 2);
    targetNoGo = true(nR, nC);                   % moves off the grid are blocked too
    rows = max(1, 1-dr) : min(nR, nR-dr);
    cols = max(1, 1-dc) : min(nC, nC-dc);
    targetNoGo(rows, cols) = noGo(rows+dr, cols+dc);

    tk = timeH(:, :, k);   tk(targetNoGo) = Inf;   timeH(:, :, k)   = tk;
    ek = effortJ(:, :, k); ek(targetNoGo) = Inf;   effortJ(:, :, k) = ek;
end

fprintf('\n--- Part B: graph costs ---\n');
fprintf('No-go cells (slope > %d deg): %.1f%% of area\n', p.maxCellSlope, 100*mean(noGo(:)));
fprintf('Blocked moves: %.1f%%\n', 100*mean(isinf(timeH(:))));

% Edge-level relationship between time and effort (uphill, passable moves only)
m = isfinite(timeH) & isfinite(effortJ) & dz > 0;
cc = corrcoef(timeH(m), effortJ(m));
fprintf('Edge-level correlation, time vs effort (uphill moves): r = %.3f\n', cc(1,2));
fprintf('  (High is expected per move - the real test is ROUTE-level correlation after A*.)\n');

% Scatter of a random subsample, coloured by grade
idx = find(m);
idx = idx(randperm(numel(idx), min(20000, numel(idx))));
figure('Name', 'Per-move costs');
scatter(timeH(idx)*3600, effortJ(idx)*p.mass/1000, 4, edgeGrade(idx), 'filled');
cb = colorbar; cb.Label.String = 'Grade of move (deg)';
xlabel('Time per move (s)'); ylabel('Effort per move (kJ)'); grid on;
title('Time vs effort for every uphill move');

%% Save
save(fullfile(dataDir, 'costs.mat'), 'timeH', 'effortJ', 'noGo', 'p', 'dirs', 'edgeLen', 'R', 'cs');
fprintf('Saved costs.mat\n');

%% ===================== Model functions =====================
function t = timeCost(Lh, dz, p)
% Time (hours) to move horizontal distance Lh while gaining dz, on skins.
%  - Downhill/flat moves: horizontal Munter term only (skins on, ascent route).
%  - Uphill at or below maxSkinAngle: Munter horizontal + vertical terms.
%  - Uphill steeper than maxSkinAngle: zigzag. The track climbs at maxSkinAngle,
%    so it needs MORE horizontal distance (dz / tan(maxSkinAngle)), plus kick turns.
    Lh = Lh .* ones(size(dz));
    t  = NaN(size(dz));
    grade = atand(dz ./ Lh);

    down = dz <= 0;
    t(down) = (Lh(down)/1000/p.kmPerUnit) / p.unitsPerHour;

    up = dz > 0 & grade <= p.maxSkinAngle;
    t(up) = (Lh(up)/1000/p.kmPerUnit + dz(up)/p.vertPerUnit) / p.unitsPerHour;

    zz = dz > 0 & grade > p.maxSkinAngle;
    LhZig = dz(zz) / tand(p.maxSkinAngle);
    nKick = dz(zz) / p.kickTurnEvery;
    t(zz) = (LhZig/1000/p.kmPerUnit + dz(zz)/p.vertPerUnit) / p.unitsPerHour ...
            + nKick * p.kickTurnTime / 3600;
end

function E = effortCost(Lh, dz, p)
% Effort (J/kg) = path length x metabolic cost per metre (Minetti et al., 2002).
% When zigzagging, the track climbs at maxSkinAngle, so we use that grade and
% the longer zigzag path length.
    Lh = Lh .* ones(size(dz));
    grade = atand(dz ./ Lh);

    i  = dz ./ Lh;                 % gradient (rise/run)
    Lp = hypot(Lh, dz);            % path length along the slope

    zz = dz > 0 & grade > p.maxSkinAngle;
    i(zz)  = tand(p.maxSkinAngle);
    Lp(zz) = dz(zz) / sind(p.maxSkinAngle);

    i = max(min(i, 0.45), -0.45);  % stay within the range Minetti measured
    E = Lp .* minettiCw(i);
    E(isnan(dz)) = NaN;
end

function C = minettiCw(i)
% Energy cost of walking (J per kg per metre) vs gradient i,
% Minetti et al. (2002), J. Appl. Physiol. 93:1039-1046. Valid for -0.45 <= i <= 0.45.
    C = 280.5*i.^5 - 58.7*i.^4 - 76.8*i.^3 + 51.9*i.^2 + 19.6*i + 2.5;
end
