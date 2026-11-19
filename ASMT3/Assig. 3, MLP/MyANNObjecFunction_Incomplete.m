function [J, GradJ] = MyANNObjecFunction_Incomplete(ANN_Weights, ...
    NumFeatures, ...
    hiddenLayerSize, ...
    NumClasses, ...
    x, t, lambda)
% This function implements the "multi-layer perceptron" objective function
% and gradient for a two layer MLP.

%%% PHASE 0:
% MLP parameters are "unrolled" into the vector "ANN_Weights" and need
% to be converted back into the weight matrices.
theta1 = reshape(ANN_Weights(1:hiddenLayerSize * (NumFeatures + 1)), ...
    hiddenLayerSize, (NumFeatures + 1));

theta2 = reshape(ANN_Weights((1 + (hiddenLayerSize * (NumFeatures + 1))):end), ...
    NumClasses, (hiddenLayerSize + 1));

% Number of Training Observations
M = size(x, 2);

J = 0;
Grad_theta1 = zeros(size(theta1));
Grad_theta2 = zeros(size(theta2));

% ********************* Write You Code Here *********************

%%% PHASE 1: Forward Propagation (all M observations at once)
% Calculate Output of First Layer
x  = [ones(1, M); x];                 % augmented input: (NumFeatures+1) x M
z2 = theta1 * x;                      % hiddenLayerSize x M
a2 = sigmf(z2, [1 1]);                % hidden layer outputs

% Calculate Output of Second Layer
a2 = [ones(1, M); a2];                % augment with bias row: (H+1) x M
z3 = theta2 * a2;                     % NumClasses x M
a3 = sigmf(z3, [1 1]);                % network outputs

% Assign a3 as the output of the network
f = a3;
f = min(max(f, 1e-10), 1 - 1e-10);   % avoid log(0) -> NaN

% Cross-entropy (logistic regression) objective
J = 0;
for i = 1:M
    J = J + sum( t(:,i) .* log(f(:,i)) + (1 - t(:,i)) .* log(1 - f(:,i)) );
end
J = -1/M * J;

% Add Regularization Term (bias columns excluded)
J = J + lambda/(2*M) * ( sum(sum(theta1(:,2:end).^2)) + ...
                         sum(sum(theta2(:,2:end).^2)) );

%%% PHASE 2: Back Propagation
Delta1 = zeros(size(theta1));
Delta2 = zeros(size(theta2));
for i = 1:M
    % Forward pass for observation i (x is already augmented)
    a1 = x(:,i);                          % (NumFeatures+1) x 1
    z2 = theta1 * a1;                     % H x 1
    a2 = sigmf(z2, [1 1]);                % H x 1

    a2 = [1; a2];                         % (H+1) x 1
    z3 = theta2 * a2;                     % NumClasses x 1
    a3 = sigmf(z3, [1 1]);                % NumClasses x 1

    % Error in output layer
    delta3 = a3 - t(:,i);                 % NumClasses x 1

    % Error in hidden layer (bias column of theta2 dropped)
    delta2 = (theta2(:,2:end)' * delta3) .* a2(2:end) .* (1 - a2(2:end));  % H x 1

    % Accumulate gradients
    Delta1 = Delta1 + delta2 * a1';       % H x (NumFeatures+1)
    Delta2 = Delta2 + delta3 * a2';       % NumClasses x (H+1)
end

% Normalize Gradients by Number of Observations
Grad_theta1 = 1/M * Delta1;
Grad_theta2 = 1/M * Delta2;

% Add Regularization to Gradients (not to bias column)
Grad_theta1(:,2:end) = Grad_theta1(:,2:end) + lambda/M * theta1(:,2:end);
Grad_theta2(:,2:end) = Grad_theta2(:,2:end) + lambda/M * theta2(:,2:end);

% *****************************************************************

% Unroll gradients
GradJ = [Grad_theta1(:) ; Grad_theta2(:)];

end
