function [ObjValues, AddOutput] = FeatureSelectionObjective( ...
    SelectedFeatures, Data, WeightTrain, NumExecutions)
% This Function Calculates the Objective Value for Multi-Objecitve Feature
% Selection Problem; 
% Objective 1: Number of Selected Features (between 0 and 1)
% Objective 2: Weighted Sum of Train Error and Test Error)

% SelectedFeatures: Binary Vector (0s and 1s) Indicating Selected Features
% e.g.,: SelectedFeatures = [1 0 1]: first and third features were selected


% Identify the Selected Features
S = find(SelectedFeatures ~= 0);

% Extracting Selecting Features
InputSelected = Data.Input(S,:);

% Number of Selected Features
NumFeature = numel(S);

% Ratio of Selected Features
RatioSelectedFeature = NumFeature/numel(SelectedFeatures);

% Weights of Train and Test Errors
WeightTest = 1 - WeightTrain;

% Multiple Executions of ANN
Error = zeros(1, NumExecutions);
for i = 1:NumExecutions
    % Create and Train ANN
    Results = TrainMyANN(InputSelected, Data.Output);
    % Calculate Weighted Sum of Train and Test Error
    Error(i) = WeightTrain * Results.TrainData.Perf + WeightTest * Results.TestData.Perf;
end

AverageError = mean(Error);
if isinf(AverageError)
    AverageError = 100;
end

% Return the Objective Function Values
ObjValues = [NumFeature; AverageError];

% Return Additional Outputs
AddOutput.S = S;
AddOutput.NumFeature = NumFeature;
AddOutput.RatioSelectedFeature = RatioSelectedFeature;
AddOutput.AverageError = AverageError;
AddOutput.ObjValues = ObjValues;
end