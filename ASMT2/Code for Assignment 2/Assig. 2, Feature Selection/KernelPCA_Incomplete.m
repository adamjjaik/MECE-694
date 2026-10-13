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
%%% You Should Complete This Function
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

switch Kernel.Type
    case 'Linear'
        K = (X*X').^Kernel.Parameter;
        
    case 'Polynomial' % Write Code for Polynomial Kernel
        % K = ;
        
    case 'Gaussian'
        sx = sum(X.^2, 2);
        sy = sum(X.^2, 2);
        xy = 2*(X*X');
        K = exp(((xy-sx)-sy')./Kernel.Parameter^2);
        
    case 'HypTan' % Write Code for Hyperbolic Tangent Kernel
        % K = ;
        
    case 'Laplacian' % Write Code for Laplacian Kernel
        % K = ;
        
    case 'RationalQuadratic' % Write Code for Rational Quadratic Kernel
        % K = ;
        
    case 'MultiQuadric' % Write Code for MultiQuadric Kernel
        % K = ;
        
    case 'Power' % Write Code for Power Kernel
        % K = ;
        
    case 'Log' % Write Code for Log Kernel   
    % K = ;
    
end


end










