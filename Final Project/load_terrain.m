%% load_terrain.m
% MECE 694 Final Project - Ski-touring route optimization, Rogers Pass
% Step 1: Load the clipped MRDEM terrain (DTM) and surface (DSM) models,
% check that the two grids line up, and take a first look at the data.
%
% Requires: Mapping Toolbox (readgeoraster, georasterinfo, projfwd, worldToDiscrete)

clear; clc; close all;

%% 1. File locations - edit these to match your folder
dataDir = 'data';
dtmFile = fullfile(dataDir, 'RogersPass_dtm.tif');
dsmFile = fullfile(dataDir, 'RogersPass_dsm.tif');

%% 2. Read the GeoTIFFs
% readgeoraster returns two things:
%   Z : the elevation values as a plain matrix (rows x cols)
%   R : a "raster reference" object that says where that matrix sits on Earth
[dtm, R]    = readgeoraster(dtmFile, 'OutputType', 'double');
[dsm, Rdsm] = readgeoraster(dsmFile, 'OutputType', 'double');

%% 3. Replace NoData cells with NaN
% We told QGIS to write -9999 for empty cells. If left in, one -9999 cell
% next to a 2000 m peak looks like an 11 km cliff to the slope calculation.
infoDTM = georasterinfo(dtmFile);
infoDSM = georasterinfo(dsmFile);
dtm = standardizeMissing(dtm, infoDTM.MissingDataIndicator);
dsm = standardizeMissing(dsm, infoDSM.MissingDataIndicator);

%% 4. Check the two grids line up (same size, extent, cell size)
% Exact equality is too strict: QGIS rounds the extent when copying it
% between layers, so corners can differ by tiny fractions of a metre.
% We use a tolerance instead.
assert(isequal(size(dtm), size(dsm)), 'DTM and DSM are different sizes');

offset = max(abs([R.XWorldLimits - Rdsm.XWorldLimits, ...
                  R.YWorldLimits - Rdsm.YWorldLimits]));   % worst corner offset (m)
fprintf('Max DTM/DSM corner offset: %.4f m\n', offset);

tol = 0.01 * R.CellExtentInWorldX;   % 1% of a cell = 0.3 m, negligible for a 30 m grid
if offset > tol
    % Genuine misalignment: resample the DSM onto the DTM grid so that
    % cell (i,j) means the same patch of ground in both layers.
    warning('Grids offset by %.2f m - resampling DSM onto the DTM grid.', offset);
    [Xq, Yq] = worldGrid(R);                       % DTM cell-centre coordinates
    dsm  = mapinterp(dsm, Rdsm, Xq, Yq, 'linear'); % DSM values at those points
    Rdsm = R;
end

%% 5. Print a summary so you can sanity-check the data
[nRows, nCols] = size(dtm);
cellSize = R.CellExtentInWorldX;   % metres (because the CRS is UTM)

fprintf('--- Terrain grid summary ---\n');
fprintf('CRS:            %s\n', R.ProjectedCRS.Name);
fprintf('Grid size:      %d rows x %d cols (%d cells)\n', nRows, nCols, numel(dtm));
fprintf('Cell size:      %.1f m x %.1f m\n', R.CellExtentInWorldX, R.CellExtentInWorldY);
fprintf('Area covered:   %.1f km (E-W) x %.1f km (N-S)\n', ...
        nCols*cellSize/1000, nRows*cellSize/1000);
fprintf('Easting range:  %.0f to %.0f m\n', R.XWorldLimits);
fprintf('Northing range: %.0f to %.0f m\n', R.YWorldLimits);
fprintf('Row 1 is the:   %s edge\n', R.ColumnsStartFrom);
fprintf('DTM elevation:  %.0f to %.0f m\n', min(dtm(:)), max(dtm(:)));
fprintf('NoData cells:   %d in DTM, %d in DSM\n', nnz(isnan(dtm)), nnz(isnan(dsm)));

%% 6. Canopy height = surface - ground
canopy = dsm - dtm;
fprintf('Canopy height:  median %.1f m, max %.1f m, %.1f%% of cells negative\n', ...
        median(canopy(:), 'omitnan'), max(canopy(:)), 100*mean(canopy(:) < 0));
% Small negative values are noise from the two models; clamp them to zero.
canopy(canopy < 0) = 0;

%% 7. Sanity check: look up the elevation at the Rogers Pass summit
% Convert lat/lon -> UTM easting/northing -> matrix row/col -> elevation.
latPass = 51.30;  lonPass = -117.52;          % approximate highway summit
[xPass, yPass] = projfwd(R.ProjectedCRS, latPass, lonPass);
[rowPass, colPass] = worldToDiscrete(R, xPass, yPass);
fprintf('Rogers Pass summit: row %d, col %d, DTM = %.0f m (expect roughly 1300-1350 m)\n', ...
        rowPass, colPass, dtm(rowPass, colPass));

%% 8. Plots
% Cell-centre coordinates of the first/last cells (imagesc wants centres, not edges)
xCentres = R.XWorldLimits + [0.5 -0.5]*cellSize;
yCentres = R.YWorldLimits + [0.5 -0.5]*cellSize;
if strcmp(R.ColumnsStartFrom, 'north')
    yCentres = fliplr(yCentres);   % row 1 is at the top (north)
end

figure('Name', 'Rogers Pass terrain', 'Position', [100 100 1400 450]);
tiledlayout(1, 3, 'TileSpacing', 'compact');

nexttile;
imagesc(xCentres, yCentres, dtm); set(gca, 'YDir', 'normal'); axis image;
colorbar; title('DTM - ground elevation (m)'); xlabel('Easting (m)'); ylabel('Northing (m)');
hold on; plot(xPass, yPass, 'r^', 'MarkerFaceColor', 'r'); hold off;

nexttile;
imagesc(xCentres, yCentres, dsm); set(gca, 'YDir', 'normal'); axis image;
colorbar; title('DSM - surface elevation (m)'); xlabel('Easting (m)');

nexttile;
imagesc(xCentres, yCentres, canopy); set(gca, 'YDir', 'normal'); axis image;
colorbar; clim([0 40]); title('Canopy height = DSM - DTM (m)'); xlabel('Easting (m)');

% 3D view of the ground surface
[X, Y] = worldGrid(R);
figure('Name', 'Rogers Pass 3D');
surf(X, Y, dtm, 'EdgeColor', 'none');
daspect([1 1 1]); view(-30, 45); camlight; lighting gouraud;
colorbar; title('Rogers Pass DTM (true vertical scale)');
xlabel('Easting (m)'); ylabel('Northing (m)'); zlabel('Elevation (m)');

%% 9. Save everything in one .mat file for the later steps
save(fullfile(dataDir, 'terrain.mat'), 'dtm', 'dsm', 'canopy', 'R', 'cellSize');
fprintf('Saved terrain.mat\n');
