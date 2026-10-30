MEC E 694 - Assignment 2: Feature Selection & Dimension Reduction
Adam Jaikaran (ID# 1708948)
======================================================================

REQUIREMENTS
------------
- MATLAB R2020a or newer
- Deep Learning Toolbox                 (bodyfat_dataset, fitnet, train)
- Statistics and Machine Learning Toolbox (sequentialfs, fsulaplacian, pca,
                                           tsne, knnsearch, pdist)

All files should be kept in the same folder. No file paths need to be edited.


======================================================================
PARTS 1-2: CLASSICAL FEATURE SELECTION AND DIMENSION REDUCTION (BodyFat)
======================================================================

Files
-----
FeatureSelection.m               Main script. Section 7 is a self-contained
                                 comparison of all methods (errors, runtimes,
                                 d = 2 scatter plots).
BestInd_d_Features_Incomplete.m  Q1 (completed): Best Individual d Features.
KernelPCA_Incomplete.m           Q2 (completed): Kernel PCA with Linear,
                                 Polynomial, Gaussian, HypTan, Laplacian,
                                 RationalQuadratic, MultiQuadric, Power and
                                 Log kernels.
FeatureSelectionObjective.m      Provided: ANN objective (weighted train/test MSE).
SequentialFeatureSelection.m     Provided: ANN objective for sequentialfs.
LoadData.m                       Provided: loads the BodyFat dataset.
KernelTable.m                    Kernel parameter study: every kernel at three
                                 parameter values (d = 3).
KernelFigures.m                  Produces report Figures 1 and 2 from the
                                 KernelTable.m results.

How to run
----------
1. Open FeatureSelection.m.
2. Run Section 1 (loads the BodyFat dataset).
3. Run Section 7 (Run Section, or Ctrl+Enter inside it). Sections 2-6 are
   the original demonstrations and do not need to be run first.
   Output: error and runtime tables (report Tables 1 and 2), the d = 2
   best/worst scatter plot (report Figure 3), and Q3_Results.mat.
   Runtime: roughly 10-20 minutes, mostly ANN training (sequential search
   and Best Individual d Features). For a quick test, set NumExecSelect and
   NumExecEval to 2 in Section 7a.
4. Report Table 3 (PCA cumulative variance) comes from Section 5.
5. Kernel study (report Figures 1 and 2):
   - Run KernelTable.m   (creates KernelTable_Results.mat)
   - Run KernelFigures.m (creates the figures; several minutes)

Settings
--------
- Random seed: rng(1) for all comparisons (rng(21:25) for the five-seed
  re-evaluation in KernelFigures.m).
- ANN: 10 hidden neurons, Levenberg-Marquardt, 70/15/15 random split.
- Reported error: mean of 0.5 x train MSE + 0.5 x test MSE over 50 runs.


======================================================================
PART 4: FEATURE SELECTION & DIMENSION REDUCTION PIPELINE (UCI HAR)
======================================================================

Files
-----
HAR_Pipeline.m               Main script: CONFIG block + all pipeline stages.
LoadHAR.m                    Loads and combines the train and test partitions.
RunKPCA.m                    Kernel PCA (Linear, Gaussian, Laplacian, Polynomial).
NeighborhoodPreservation.m   Fraction of k nearest neighbours preserved in an
                             embedding (unsupervised quality metric).
PlotEmbedding2D.m            2D scatter coloured by activity.
PlotEmbedding3D.m            3D scatter coloured by activity.

Data setup (required before running)
------------------------------------
1. Download "Human Activity Recognition Using Smartphones" from the UCI
   Machine Learning Repository (dataset 240).
2. The download is a zip containing another zip ("UCI HAR Dataset.zip").
   Extract BOTH levels.
3. Place the folder "UCI HAR Dataset" (the folder that directly contains
   features.txt, activity_labels.txt, train/ and test/) next to
   HAR_Pipeline.m:

       <code folder>/
           HAR_Pipeline.m
           LoadHAR.m
           ...
           UCI HAR Dataset/
               features.txt
               activity_labels.txt
               train/
               test/

   If the folder is missing, LoadHAR.m stops with a message explaining
   where to put it.

How to run
----------
Open HAR_Pipeline.m and press Run. The full pipeline takes about 30 s
(t-SNE is the slowest step at about 3 s). Paths are built from the script's
own location, so no editing is needed.

Pipeline stages
---------------
1. Data ingestion & sanity checks (samples, features, missing values,
   range, constant features, outliers, label distribution)
2. Preprocessing: z-score standardization (parameters saved)
3. Feature selection: Laplacian Score
4. Dimension reduction: PCA (full data), Kernel PCA (Gaussian and
   Laplacian kernels), t-SNE
5. Visualization and quantitative comparison (neighbourhood preservation,
   runtime)

Settings (CONFIG block at the top of HAR_Pipeline.m)
---------------------------------------------------
Config.Seed     = 1       random seed (subsample and t-SNE)
Config.NSub     = 3000    balanced subsample, 500 per activity (needed
                          because Kernel PCA builds an n x n matrix)
Config.KPCA     Gaussian (sigma = 0.5 x median distance) and
                Laplacian (sigma = 1.0 x median distance)
Config.RunTSNE  = true    set to false to skip t-SNE
Config.kNP      = 10      neighbours for the preservation metric

Outputs (saved automatically to results/)
-----------------------------------------
LabelDistribution.csv       activity counts            (report Table 4)
Preprocessing.mat           standardization parameters
LaplacianScores.png         sorted Laplacian Scores
PCA_ExplainedVariance.png   cumulative explained variance
KPCA_ParameterSweep.csv     neighbourhood preservation vs. kernel parameter
KPCA_ParameterCheck.png     KPCA embeddings at several parameters
Embeddings_2D.png           2D embeddings of all methods (report Figure 4)
Embeddings_3D.png           3D embeddings, PCA vs. KPCA  (report Figure 5)
MethodComparison.csv        neighbourhood preservation and runtime
                            (report Table 5)

Note: Activity labels are used only to colour the plots. They are not used
for feature selection or parameter tuning.
