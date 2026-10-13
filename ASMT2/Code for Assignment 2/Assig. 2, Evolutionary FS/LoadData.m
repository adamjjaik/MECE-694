function Data = LoadData(Dataset_Name)
% This Function Loads a Dataset For Feature Selection or
% Dimension Reduction with Supervised Learning Algorithms

switch Dataset_Name
    case 'BodyFat'
        % 252 Inputs/Observations (13 Features)
        % 1 Output
        [Input, Output] = bodyfat_dataset();
        
    case 'BreastCancer'
        % 699 Inputs/Observations (6 Features)
        % 2 Output
        [Input, Output] = cancer_dataset();    
end

% Save All Infromation In Structure Data
Data.Input = Input;
Data.Output = Output;
Data.NumFeatures = size(Input,1);
Data.NumOutputs = size(Output,1);
Data.NumObservations = size(Input,2);
end