function [Index, Error] = BestInd_d_Features_Incomplete(d, Objective, Data)
% This Function Performs Best Individual d Feature Selection
% Complete This Function To Select Features Using Individual d Feature Selection

% Outputs are Index and Error
% (1) Index is a 1 by Number of Features vector which contains 1s for selected
% d Features and 0s for Other Features

% (2) Error Contains the Weighted Train + Test Error Obtained By Using only
% One Feature. Error is a 1 by Number of Features vector.

% Objective is the Objective Function Handle Which You Can Call to
% Calculate Error for Each Feature. Objective Input is a Vector Similar to
% Index.


Index = []; % You Should Find/Return Index Values
Error = []; % You Should Find/Return Error Values

% STEP 1:
% Extract Input Data
Input = Data.Input;

% Evaluate ANN Error Using One Feature at a Time
NumFeatures = size(Input, 1);
Error = zeros(1, NumFeatures);
for j = 1:NumFeatures
    Selected = zeros(1, NumFeatures); % binary vector
    Selected(j) = 1; % only j-th element = 1
    % Train/test the ANN with this single feature
    % Objective returns a scalar error
    Error(j) = Objective(Selected); 
end

% STEP 2: Sort Obtained Errors
% Sort from lowest to highest error,
% order(1) is the feature with the lowest error,
% use "~" to ignore the sorted values so Error stays
% indexed by feature number as the function header requires
[~, order] = sort(Error, 'ascend');


% STEP 3: 
% Select First d Features With Minimum Error
Index = zeros(1, NumFeatures);

% Create a Binary output Named "Index"
Index(order(1:d)) = 1;

end




