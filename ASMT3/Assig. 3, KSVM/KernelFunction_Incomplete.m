function [K] = KernelFunction_Incomplete(Kernel)
% This function retuns function handle to various kernel functions to 
% map data to a higher dimension 

% Kernel.Type: Name of the kernel function including Linear, Gaussian,
% Logistic, HypTan

% Kernel.Parameter: Parameter(s) associated with each kernel function
%   Linear   : beta            -> (1 + xi'*xj)^beta
%   Gaussian : sigma           -> exp(-||xi - xj||^2 / (2*sigma^2))
%   Logistic : [beta0 beta1]   -> 1 / (1 + exp(-(beta0 + beta1*xi'*xj)))
%   HypTan   : [beta0 beta1]   -> tanh(beta0 + beta1*xi'*xj)
%   (For Logistic and HypTan, a single value is treated as beta1 with beta0 = 0)

% K: function handle for a user specified kernel function
% K operates on two inputs xi and xj

% Provide the code to compute Gaussian, Logistic, HypTan kernels
% Linear kernel function is provided as an example for you

% ************** Write Your Code Here **********************
switch Kernel.Type
    case 'Linear'
        K = @(xi, xj) (1+xi'*xj)^Kernel.Parameter;
        
    case 'Gaussian'
        sigma = Kernel.Parameter;
        K = @(xi, xj) exp(-sum((xi - xj).^2) / (2*sigma^2));
        
    case 'Logistic'
        [beta0, beta1] = GetBetas(Kernel.Parameter);
        K = @(xi, xj) 1 / (1 + exp(-(beta0 + beta1*(xi'*xj))));
        
    case 'HypTan'
        [beta0, beta1] = GetBetas(Kernel.Parameter);
        K = @(xi, xj) tanh(beta0 + beta1*(xi'*xj));
        
end
% **********************************************************
end

function [beta0, beta1] = GetBetas(Parameter)
% Helper: allows Parameter = beta1 (scalar) or Parameter = [beta0 beta1]
if numel(Parameter) == 1
    beta0 = 0;
    beta1 = Parameter;
else
    beta0 = Parameter(1);
    beta1 = Parameter(2);
end
end