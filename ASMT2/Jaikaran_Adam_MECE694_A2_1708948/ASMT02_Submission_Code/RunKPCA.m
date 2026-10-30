function Z = RunKPCA(X, KernelType, Param, d)
% RunKPCA  Kernel PCA embedding of the rows of X.
%   Z = RunKPCA(X, KernelType, Param, d) returns an n x d matrix of the first
%   d kernel principal components.
%   X          n x p data (rows = observations), already standardized
%   KernelType 'Linear' | 'Gaussian' | 'Laplacian' | 'Polynomial'
%   Param      Gaussian/Laplacian: bandwidth sigma; Polynomial: degree
%   d          number of components to return

p   = size(X, 2);
G   = X * X';                                   % n x n dot products
sq  = diag(G);
D2  = max(sq + sq' - 2*G, 0);                   % squared distances (clip rounding negatives)

switch KernelType
    case 'Linear'       % equivalent to PCA (sanity check)
        K = G;
    case 'Gaussian'     % exp(-||x-y||^2 / (2 sigma^2))
        K = exp(-D2 / (2*Param^2));
    case 'Laplacian'    % exp(-||x-y|| / sigma)
        K = exp(-sqrt(D2) / Param);
    case 'Polynomial'   % (x'y/p + 1)^degree; dividing by p keeps values moderate
        K = (G/p + 1).^Param;
    otherwise
        error('Unknown kernel type: %s', KernelType);
end

% Centre the kernel matrix in feature space (equivalent to subtracting the mean)
Kc = K - mean(K, 1) - mean(K, 2) + mean(K(:));
Kc = (Kc + Kc') / 2;                             % enforce exact symmetry

% Leading d eigenvectors only (much cheaper than a full decomposition)
[V, L]    = eigs(Kc, d, 'largestreal');
[lam, ix] = sort(diag(L), 'descend');
V         = V(:, ix);

% Projections of the training points onto the kernel principal components
Z = V .* sqrt(max(lam, 0))';
end