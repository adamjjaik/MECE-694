# MECE 694 Assignment 3, Part 4: From FS/DR to Classification & Clustering
Adam Jaikaran (ID 1708948)

## Requirements
MATLAB R2023b or newer with the Statistics and Machine Learning Toolbox and the Deep Learning Toolbox.

## Setup
Put the extracted `UCI HAR Dataset` folder in the same folder as the scripts.

## How to run
1. `HAR_Classification.m`: split, scaling, PCA (4.1) and classification study (4.2). Takes about 8 min.
2. `HAR_Clustering.m`: k-means (4.3). Run after step 1, since it loads the saved PCA data.

All settings are in the CONFIG block at the top of each script (seed = 1). Figures and tables are saved to `results_A3/`.

## Files
| File | Purpose |
|---|---|
| `LoadHAR.m` | Loads the HAR dataset (train + test combined) |
| `StratifiedSplit.m` | Stratified 60/20/20 train/val/test split |
| `FitScaler.m`, `ApplyScaler.m` | Z-score scaler (fit on train only) |
| `FitPCA.m`, `ApplyPCA.m` | PCA keeping 95% variance (fit on train only) |
| `DefineClassifiers.m` | Classifiers and the parameter settings tried |
| `CVScore.m` | 3-fold cross-validation routine |
| `ClassMetrics.m` | Accuracy, sensitivity, specificity |
| `MatchClusters.m` | Matches k-means clusters to activities for scoring |
