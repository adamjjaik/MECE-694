%% Reference
% Part of this code was adopted from: 
% https://www.mathworks.com/matlabcentral/fileexchange/69378-kernel-principal-component-analysis-kpca

% This code was modifies for "MECE 788: Applied Computational Intelligence
% for Engineering" at the Univerisyt of Alberta, Winter 2021.


function [MappedData]= KernelPCA_Incomplete(Input, Kernel, d)
% This Function Performs Kernel PCA On Input Data
% Input Data: M by p Matrix (M: # of Observations, p: # of Features)
% Kernel: Structure Including "Type" and "Parameter"
% "Type" is Kernel Name: Linear, Polynomial, Gaussian, etc
% "Parameter" is Kernel Parameter for Each Kernel Type

% Compute the Kernel Matrix
K = ComputeKernelMatrix(Input, Kernel); 

% Center Kernel Matrix
OneMatrix = ones(size(K))./size(K,1);
K_center = K - OneMatrix * K - K * OneMatrix + OneMatrix * K * OneMatrix;

% Perform Eigenvalue Decomposition
[EigVec, EigVal] = eigs(K_center/size(K_center,1), size(K_center,1));
lambda = diag(EigVal);

% Sort EigenVectors Using EigenValues
[~, index] = sort(lambda, 'descend');
EigVec = EigVec(:, index);

% Calculate The Mapped Data
MappedData = EigVec(:,1:d)' * K_center;
end

function [K] = ComputeKernelMatrix(X, Kernel)
% This Function Calculates the Kernel Matrix For Input Data X
% More Kernel Functions Can be found at:
% http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#linear

% Input Data: M by p Matrix (M: # of Observations, p: # of Features)
% Kernel: Structure Including "Type" and "Parameter"


% computing once to be used in cases below
sx = sum(X.^2, 2);             % M by 1 (squared norm of each observation)
Dist2 = sx + sx' - 2*(X*X');   % M by M matrix of squared distances
Dist2 = max(Dist2, 0);         % rounding error can create tiny negatives
Dist  = sqrt(Dist2);           % M by M matrix of distances


switch Kernel.Type
    case 'Linear'
        K = (X*X').^Kernel.Parameter;
        
    case 'Polynomial' % K = (xi'*xj + c)^d, with c = 1 and degree d = Kernel.Parameter
        K = (X*X' + 1).^Kernel.Parameter;
        
    case 'Gaussian'
        sx = sum(X.^2, 2);
        sy = sum(X.^2, 2);
        xy = 2*(X*X');
        K = exp(((xy-sx)-sy')./Kernel.Parameter^2);
        
    case 'HypTan' % K = tanh(alpha*xi'*xj + c), with alpha = Kernel.Parameter and c = 1
        K = tanh(Kernel.Parameter*(X*X') + 1);
        
    case 'Laplacian' % K = exp(-||xi - xj|| / sigma), with sigma = Kernel.Parameter
        K = exp(-Dist./Kernel.Parameter);
        
    case 'RationalQuadratic' % K = 1 - ||xi - xj||^2 / (||xi - xj||^2 + c), with c = Kernel.Parameter
       K = 1 - Dist2./(Dist2 + Kernel.Parameter);
        
    case 'MultiQuadric' % K = sqrt(||xi - xj||^2 + c^2), with c = Kernel.Parameter
        K = sqrt(Dist2 + Kernel.Parameter^2);
        
    case 'Power' % K = -||xi - xj||^d, with d = Kernel.Parameter
        K = -Dist.^Kernel.Parameter;
        
    case 'Log' % K = -log(||xi - xj||^d + 1), with d = Kernel.Parameter   
        K = -log(Dist.^Kernel.Parameter + 1);

    otherwise % error message if kernel type is not listed
        error('Unknown kernel type: %s', Kernel.Type);
    
end


end










