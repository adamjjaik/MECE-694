function HAR = LoadHAR(dataDir)
% LoadHAR  Load the UCI Human Activity Recognition dataset (train + test combined).
%   HAR = LoadHAR(dataDir) reads the raw text files from the "UCI HAR Dataset"
%   folder and returns a structure with:
%     X             n x p feature matrix (rows = samples, columns = features)
%     y             n x 1 activity label (1..6)
%     Subject       n x 1 subject ID (1..30)
%     Split         n x 1 categorical, 'train' or 'test' (original split)
%     FeatureNames  p x 1 cell array of feature names
%     ActivityNames 6 x 1 cell array of activity names
%   Note: unlike Data.Input in Parts 1-3, rows here are observations, which is
%   the convention used by MATLAB's pca, tsne and fsulaplacian.

% Feature names: each line of features.txt is "<index> <name>"
fid = fopen(fullfile(dataDir, 'features.txt'));
C = textscan(fid, '%d %s');  fclose(fid);
FeatureNames = C{2};

% Activity names: each line of activity_labels.txt is "<id> <name>"
fid = fopen(fullfile(dataDir, 'activity_labels.txt'));
C = textscan(fid, '%d %s');  fclose(fid);
ActivityNames = C{2};

% Numeric data (whitespace-delimited text files)
Xtr = load(fullfile(dataDir, 'train', 'X_train.txt'));
ytr = load(fullfile(dataDir, 'train', 'y_train.txt'));
str = load(fullfile(dataDir, 'train', 'subject_train.txt'));
Xte = load(fullfile(dataDir, 'test',  'X_test.txt'));
yte = load(fullfile(dataDir, 'test',  'y_test.txt'));
ste = load(fullfile(dataDir, 'test',  'subject_test.txt'));

% Combine train + test, as the assignment requires
HAR.X             = [Xtr; Xte];
HAR.y             = [ytr; yte];
HAR.Subject       = [str; ste];
HAR.Split         = categorical([repmat({'train'}, numel(ytr), 1); repmat({'test'}, numel(yte), 1)]);
HAR.FeatureNames  = FeatureNames;
HAR.ActivityNames = ActivityNames;
end