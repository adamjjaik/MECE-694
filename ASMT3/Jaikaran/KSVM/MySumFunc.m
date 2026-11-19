function [SUM] = MySumFunc(xi, alpha, yj, xj, K)
% This function calculates: Sum(alpha_j * y_j * Kernel(x_j, x_i))

SUM = 0;
for j = 1:length(alpha)
    SUM = SUM + alpha(j) * yj(j) * K(xj(:,j), xi);
end
end