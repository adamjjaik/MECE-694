function [theta_t, ObjectiveFunction, numVar, VarMin, VarMax] = CreateProblem(Function)
% This function creates nonlinear models for parameter estimation


switch Function
    case 'Function 1'
        % Create Data
        theta_t = [0.71 1.32 2.35 3.14];
        x = [-30:0.2:30]'; %#ok
        y = theta_t(1) + theta_t(2) .* abs(x) ...
            + theta_t(3) .* cos(x) + theta_t(4) .* sin(x) ...
            + (-1.5 + 3 * rand(length(x),1)); % adding noise to data
        
        % Define Objective Function
        ObjectiveFunction = @(theta) NonLinearRegression1(x, y, theta);
        % Number of Decision Variables
        numVar = 4;
        % Decision Variables Bound
        VarMin = -2;
        VarMax = 5;
        
        figure(1)
        scatter(x, y, 'b*')
        ylabel('y'), xlabel('x')
        axis([-20 20 -1 30])
    case 'Function 2' % DROP-WAVE FUNCTION
        % Create Data
        theta_t = [0.11 0.75 2.71 1.48];
        [X1,X2] = meshgrid(-5:0.1:5,-5:0.1:5);
        Y = theta_t(1) - (1 + theta_t(2) .* cos(12.*sqrt(X1.^2+X2.^2)))./...
            (0.5.*(theta_t(3) .* X1.^2 + theta_t(4) .* X2.^2) + 2);
        
        x = [X1(:) X2(:)];
        y = Y(:);
        
        % Define Objective Function
        ObjectiveFunction = @(theta) Drop_Wave_Function(x, y, theta);
        % Number of Decision Variables
        numVar = 4;
        % Decision Variables Bound
        VarMin = -2;
        VarMax = 5;
        
        figure(1)
        surf(X1, X2, Y)
        xlabel('x1'), ylabel('x2'), zlabel('y')
        axis([-5 5 -5 5 -1.5 0.5])
    case 'Function 3' % EASOM FUNCTION
        % Create Data
        theta_t = [0.52 3.31 1.01 1.34];
        [X1,X2] = meshgrid(-20:0.5:20,-20:0.5:20);
        Y = theta_t(1) + (-theta_t(2) .* cos(X1) .* cos(X2)) .* ...
            exp(-(theta_t(3) .* X1 - pi).^2 - (theta_t(4) .* X2 - pi).^2); 
        
        x = [X1(:) X2(:)];
        y = Y(:);
        
        % Define Objective Function
        ObjectiveFunction = @(theta) Easom_Function(x, y, theta);
        % Number of Decision Variables
        numVar = 4;
        % Decision Variables Bound
        VarMin = -2;
        VarMax = 5;
        
        figure(1)
        surf(X1, X2, Y)
        xlabel('x1'), ylabel('x2'), zlabel('y')
        axis([-20 20 -20 20 -2 1]) 
end

end


function [MSE] = NonLinearRegression1(x, y, theta)
% This function calculates the output of nonlinear function
% y = theta0 + theta1 * abs(x) + theta_2t * cos(x) + theta_3t * sin(x)
% for a given set of parameters theta


y_hat = theta(1) + theta(2) .* abs(x) ...
    + theta(3) .* cos(x) + theta(4) .* sin(x);

% Mean-squared Error
MSE = immse(y, y_hat);
end

function [MSE] = Drop_Wave_Function(x, y, theta)
% This function calculates the output of nonlinear function
% see https://www.sfu.ca/~ssurjano/drop.html for reference to this function

y_hat = theta(1) - (1 + theta(2) .* cos(12 .* sqrt(x(:,1).^2 + x(:,2).^2)))...
    ./(0.5 .* (theta(3) .* x(:,1).^2 + theta(4) .* x(:,2).^2) + 2);

% Mean-squared Error
MSE = immse(y, y_hat);
end

function [MSE] = Easom_Function(x, y, theta)
% This function calculates the output of nonlinear function
% see https://www.sfu.ca/~ssurjano/easom.html for reference to this function

y_hat =  theta(1) + (-theta(2) .* cos(x(:,1)) .* cos(x(:,2))) .* ...
            exp(-(theta(3) .* x(:,1) - pi).^2 - (theta(4) .* x(:,2) - pi).^2); 

        % Mean-squared Error
MSE = immse(y, y_hat);
end








