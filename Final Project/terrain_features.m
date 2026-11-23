%% terrain_features.m
% MECE 694 Final Project - Ski-touring route optimization, Rogers Pass
% Step 2: Compute terrain features from the DTM.
%
%   slope      - steepness of each cell (deg)              -> exposure (later)
%   aspect     - compass direction each slope faces (deg)  -> wind/lee, sun
%   curvature  - ridge (+) vs gully/terrain trap (-)       -> exposure (later)
%   hillshade  - shaded relief, used to CHECK the aspect sign
%   edgeGrade  - signed climb angle between each cell and its 8 neighbours
%                -> time and effort costs on the graph
%   elevBand   - below treeline / treeline / alpine        -> danger rating bands
%
% Input:  data/terrain.mat   (from load_terrain.m)
% Output: data/features.mat

clear; clc; close all;

dataDir = 'data';
load(fullfile(dataDir, 'terrain.mat'), 'dtm', 'R', 'cellSize');
[nR, nC] = size(dtm);
cs = cellSize;   % 30 m

% The finite-difference stencils below assume row 1 = north, column 1 = west
assert(strcmp(R.ColumnsStartFrom, 'north') && strcmp(R.RowsStartFrom, 'west'), ...
       'Grid orientation is not north-up/west-left; the stencils need flipping.');

%% 1. Slope and aspect - Horn's method
% Horn (1981) is the standard 3x3 finite-difference method used by QGIS and
% ArcGIS. It weights the direct neighbours twice as much as the diagonals,
% which makes it less sensitive to noise than a simple central difference.
%
% Neighbourhood of each cell e (row 1 = north):
%       a  b  c        NW  N  NE
%       d  e  f   =    W   e  E
%       g  h  i        SW  S  SE

Zp = dtm([1 1:end end], [1 1:end end]);       % pad edges by copying the border
a = Zp(1:end-2, 1:end-2);  b = Zp(1:end-2, 2:end-1);  c  = Zp(1:end-2, 3:end);
d = Zp(2:end-1, 1:end-2);                             f  = Zp(2:end-1, 3:end);
g = Zp(3:end,   1:end-2);  h = Zp(3:end,   2:end-1);  i3 = Zp(3:end,   3:end);

dzdx = ((c + 2*f + i3) - (a + 2*d + g)) / (8*cs);   % rise per metre going EAST
dzdy = ((a + 2*b + c) - (g + 2*h + i3)) / (8*cs);   % rise per metre going NORTH
% Note: dzdy is written "top row minus bottom row" because row 1 is north.
% Using MATLAB's gradient() here would give the opposite sign, because it
% assumes y increases with row number - the trap mentioned earlier.

slope = atand(hypot(dzdx, dzdy));                  % degrees, 0 = flat

% Aspect = direction the slope FACES = direction of steepest DESCENT,
% as a compass bearing (0 = N, 90 = E, 180 = S, 270 = W).
aspect = mod(atan2d(-dzdx, -dzdy), 360);
aspect(slope < 2) = NaN;                           % flat ground has no meaningful aspect

%% 2. Hillshade - an independent check that slope and aspect are right
% Same settings as QGIS's default hillshade: sun from the NW (315 deg), 45 deg up.
% If our aspect had the wrong sign, this image would look lit from the wrong
% side compared with the QGIS hillshade you made earlier.
sunAz = 315;  sunAlt = 45;
aspFilled = aspect;  aspFilled(isnan(aspFilled)) = 0;
hillshade = cosd(90 - sunAlt) .* cosd(slope) + ...
            sind(90 - sunAlt) .* sind(slope) .* cosd(sunAz - aspFilled);
hillshade = max(hillshade, 0);

%% 3. Curvature - ridges vs gullies
% Laplacian of a lightly smoothed DTM. Curvature from a 30 m grid is noisy,
% so we smooth over 3x3 cells (90 m) first - which is also the corridor scale
% we decided the project works at.
% Sign convention: + = convex (ridge, rollover), - = concave (gully, terrain trap).
Zs  = conv2(Zp, ones(3)/9, 'valid');               % 3x3 moving average, same size as dtm
Zsp = Zs([1 1:end end], [1 1:end end]);
lap = (Zsp(1:end-2, 2:end-1) + Zsp(3:end, 2:end-1) + ...
       Zsp(2:end-1, 1:end-2) + Zsp(2:end-1, 3:end) - 4*Zs) / cs^2;
curvature = -lap * 100;                            % units: 1/(100 m)

%% 4. Edge grades - the climb angle for each possible move
% The graph connects every cell to its 8 neighbours. For time and effort,
% what matters is the grade ALONG THE DIRECTION OF TRAVEL, not the cell's slope:
% a skin track can traverse a 35 deg slope at a gentle 10 deg climb.
%
% edgeGrade(r, c, k) = climb angle (deg) moving from cell (r,c) to its
% neighbour in direction k. Positive = uphill, negative = downhill.
% Moving the opposite way gives the same magnitude with the opposite sign,
% which is why the graph is DIRECTED.

dirs     = [-1 0; -1 1; 0 1; 1 1; 1 0; 1 -1; 0 -1; -1 -1];   % [d_row d_col]
dirNames = {'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'};      % row - 1 = one cell north
edgeLen  = cs * sqrt(sum(dirs.^2, 2));                        % 30 m straight, 42.4 m diagonal

edgeGrade = NaN(nR, nC, 8);
for k = 1:8
    dr = dirs(k, 1);  dc = dirs(k, 2);
    rows = max(1, 1-dr) : min(nR, nR-dr);      % cells whose neighbour is inside the grid
    cols = max(1, 1-dc) : min(nC, nC-dc);
    dz = dtm(rows+dr, cols+dc) - dtm(rows, cols);
    gk = NaN(nR, nC);
    gk(rows, cols) = atand(dz / edgeLen(k));
    edgeGrade(:, :, k) = gk;
end

%% 5. Elevation bands (to match the avalanche bulletin's three bands)
% PLACEHOLDER thresholds - adjust with your local knowledge of where treeline
% sits around Connaught Creek (the forest edge on the Uptrack map is a good guide).
btlTop = 1850;   % below treeline  < btlTop
tlTop  = 2150;   % treeline        btlTop..tlTop,  alpine > tlTop
elevBand = ones(nR, nC);                 % 1 = below treeline
elevBand(dtm >= btlTop) = 2;             % 2 = treeline
elevBand(dtm >= tlTop)  = 3;             % 3 = alpine

%% 6. Summary statistics
fprintf('--- Terrain features ---\n');
fprintf('Slope: median %.1f deg, max %.1f deg\n', median(slope(:)), max(slope(:)));

slopeEdges  = [0 20 25 30 35 40 45 90];
slopeLabels = {'<20', '20-25', '25-30', '30-35', '35-40', '40-45', '>45'};
counts = histcounts(slope(:), slopeEdges);
for k = 1:numel(counts)
    fprintf('  %-6s deg: %5.1f%% of area\n', slopeLabels{k}, 100*counts(k)/numel(slope));
end

% Demonstrate the "traverse" effect: on steep cells, how gentle is the
% gentlest UPHILL move available?
uphill = edgeGrade;  uphill(uphill <= 0) = NaN;          % uphill moves only
gentlestUp = min(uphill, [], 3);                         % gentlest climb from each cell
steep = slope >= 30 & slope < 35;
fprintf('On 30-35 deg cells, the gentlest uphill move has a median grade of %.1f deg\n', ...
        median(gentlestUp(steep), 'omitnan'));
fprintf('  -> traversing lets a skin track climb steep slopes at a manageable angle.\n');

fprintf('Elevation bands: BTL %.0f%%, TL %.0f%%, ALP %.0f%%\n', ...
        100*mean(elevBand(:)==1), 100*mean(elevBand(:)==2), 100*mean(elevBand(:)==3));

%% 7. Plots
% Rogers Pass summit marker, for orientation
[xPass, yPass] = projfwd(R.ProjectedCRS, 51.30, -117.52);

figure('Name', 'Terrain features', 'Position', [50 50 1400 950]);
tiledlayout(2, 2, 'TileSpacing', 'compact');

% (a) Hillshade - compare with the QGIS hillshade
nexttile;
showLayer(hillshade, R, cs, 'Hillshade (sun from NW) - should match QGIS');
colormap(gca, gray); clim([0 1]);
plotMarker(xPass, yPass);

% (b) Slope in avalanche-relevant classes
nexttile;
slopeClass = discretize(slope, slopeEdges);
showLayer(slopeClass, R, cs, 'Slope angle (deg)');
slopeCmap = [0.90 0.90 0.90;   % <20   light grey
             0.65 0.85 0.65;   % 20-25 green
             1.00 0.95 0.40;   % 25-30 yellow
             1.00 0.70 0.25;   % 30-35 orange
             0.95 0.35 0.20;   % 35-40 red-orange
             0.75 0.05 0.10;   % 40-45 red
             0.40 0.00 0.40];  % >45   purple
colormap(gca, slopeCmap); clim([0.5 7.5]);
cb = colorbar; cb.Ticks = 1:7; cb.TickLabels = slopeLabels;
plotMarker(xPass, yPass);

% (c) Aspect - circular colour map, flat cells grey
nexttile;
showLayer(aspect, R, cs, 'Aspect (direction slope faces)');
colormap(gca, hsv); clim([0 360]);
cb = colorbar; cb.Ticks = 0:90:360; cb.TickLabels = {'N', 'E', 'S', 'W', 'N'};
plotMarker(xPass, yPass);

% (d) Curvature - diverging map, symmetric limits
nexttile;
showLayer(curvature, R, cs, 'Curvature: + ridge, - gully');
v = sort(abs(curvature(:)));  lim = v(round(0.98*numel(v)));
n = 128;  t = linspace(0, 1, n)';
divCmap = [[t; ones(n,1)], [t; flipud(t)], [ones(n,1); flipud(t)]];   % blue-white-red
colormap(gca, divCmap); clim([-lim lim]); colorbar;
plotMarker(xPass, yPass);

% Slope histogram
figure('Name', 'Slope distribution');
histogram(slope, 0:2.5:70, 'Normalization', 'percentage');
xlabel('Slope angle (deg)'); ylabel('% of area'); grid on;
title('Slope distribution - Rogers Pass study area');
xline(30, '--r', 'start zones ~30-45 deg');
xline(45, '--r');

%% 8. Save
save(fullfile(dataDir, 'features.mat'), 'slope', 'aspect', 'curvature', 'hillshade', ...
     'edgeGrade', 'dirs', 'dirNames', 'edgeLen', 'elevBand', 'btlTop', 'tlTop', 'R', 'cs');
fprintf('Saved features.mat\n');

%% ---------- Local helper functions ----------
function showLayer(Z, R, cs, ttl)
    % Plot a grid layer in real-world UTM coordinates, north up.
    xC = R.XWorldLimits + [0.5 -0.5]*cs;          % first/last cell centres
    yC = fliplr(R.YWorldLimits + [0.5 -0.5]*cs);  % row 1 at the top (north)
    imagesc(xC, yC, Z, 'AlphaData', ~isnan(Z));   % NaN cells transparent
    set(gca, 'YDir', 'normal', 'Color', [0.6 0.6 0.6]);
    axis image; colorbar;
    title(ttl); xlabel('Easting (m)'); ylabel('Northing (m)');
end

function plotMarker(x, y)
    hold on; plot(x, y, 'k^', 'MarkerFaceColor', 'w', 'MarkerSize', 8); hold off;
end
