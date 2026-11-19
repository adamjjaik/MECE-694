function [x, t, NumCluster] = CreateClusteringData(Model)
% This function is not for students

switch Model
    case 'Model 1'
        % Number of Clusters
        NumCluster = 3;
        
        % Cluster 1
        Mu1 = [8 8]; Sigma1 = [2  0.1; 0.1  2];
        X1 = mvnrnd(Mu1, Sigma1, 65);
        % Cluster 2
        Mu1 = [8 -2]; Sigma1 = [1  0; 0  1];
        X2 = mvnrnd(Mu1, Sigma1, 45);
        % Cluster 3
        Mu2 = [1 1]; Sigma2 = [4  0.5; 0.5  4];
        X3 = mvnrnd(Mu2, Sigma2, 55);
        
        x = [X1; X2; X3];
        t = [1*ones(size(X1,1),1); 2*ones(size(X2,1),1); 3*ones(size(X3,1),1)];
        C = [mean(X1); mean(X2); mean(X3)];
        
    case 'Model 2'
        % Number of Clusters
        NumCluster = 3;
        
        % Cluster 1
        Mu1 = [6 6]; Sigma1 = [2  0.1; 0.1  2];
        X1 = mvnrnd(Mu1, Sigma1, 65);
        % Cluster 2
        Mu1 = [6 -2]; Sigma1 = [2  0; 0  2];
        X2 = mvnrnd(Mu1, Sigma1, 45);
        % Cluster 3
        Mu2 = [1 1]; Sigma2 = [4  0.5; 0.5  4];
        X3 = mvnrnd(Mu2, Sigma2, 55);
        
        x = [X1; X2; X3];
        t = [1*ones(size(X1,1),1); 2*ones(size(X2,1),1); 3*ones(size(X3,1),1)];
        C = [mean(X1); mean(X2); mean(X3)];
        
    case 'Model 3'
        % Number of Clusters
        NumCluster = 5;
        
        % Cluster 1
        Mu1 = [9 9]; Sigma1 = [2  0.1; 0.1  2];
        X1 = mvnrnd(Mu1, Sigma1, 65);
        % Cluster 2
        Mu1 = [7 -2]; Sigma1 = [1  0; 0  1];
        X2 = mvnrnd(Mu1, Sigma1, 45);
        % Cluster 3
        Mu2 = [-1 -1]; Sigma2 = [2  0.5; 0.5  2];
        X3 = mvnrnd(Mu2, Sigma2, 55);
        % Cluster 4
        Mu4 = [0 8]; Sigma4 = [2  0.5; 0.5  2];
        X4 = mvnrnd(Mu4, Sigma4, 55);
        % Cluster 5
        Mu5 = [4 4]; Sigma5 = [1  0.5; 0.5  1];
        X5 = mvnrnd(Mu5, Sigma5, 55);
        
        x = [X1; X2; X3; X4; X5];
        t = [1*ones(size(X1,1),1); 2*ones(size(X2,1),1); 3*ones(size(X3,1),1); ...
            4*ones(size(X4,1),1); 5*ones(size(X5,1),1)];
        C = [mean(X1); mean(X2); mean(X3); mean(X4); mean(X5)];
        
    case 'Model 4'
        % Number of Clusters
        NumCluster = 5;
        
        % Cluster 1
        Mu1 = [9 9]; Sigma1 = [2  0.1; 0.1  2];
        X1 = mvnrnd(Mu1, Sigma1, 65);
        % Cluster 2
        Mu1 = [7 -2]; Sigma1 = [3  0; 0  3];
        X2 = mvnrnd(Mu1, Sigma1, 45);
        % Cluster 3
        Mu2 = [-1 -1]; Sigma2 = [3  0.5; 0.5  3];
        X3 = mvnrnd(Mu2, Sigma2, 55);
        % Cluster 4
        Mu4 = [0 8]; Sigma4 = [4  0.5; 0.5  4];
        X4 = mvnrnd(Mu4, Sigma4, 55);
        % Cluster 5
        Mu5 = [4 4]; Sigma5 = [3  0.5; 0.5  3];
        X5 = mvnrnd(Mu5, Sigma5, 55);
        
        x = [X1; X2; X3; X4; X5];
        t = [1*ones(size(X1,1),1); 2*ones(size(X2,1),1); 3*ones(size(X3,1),1); ...
            4*ones(size(X4,1),1); 5*ones(size(X5,1),1)];
        C = [mean(X1); mean(X2); mean(X3); mean(X4); mean(X5)];
end

% Plot Data
figure(1)
% Create "NumCluster" distinct clusters
Colors = hsv(NumCluster);
% Plot data points associated with each cluster
for i = 1:NumCluster
    plot(x(t==i,1), x(t==i,2), 'p', 'LineWidth', 2, 'Color', Colors(i,:));
    hold on;
end
% Plot Cluster Centers
plot(C(:,1), C(:,2), '*k','LineWidth',2, 'MarkerSize', 12);
hold off;
ylabel('Feature: x_2'), xlabel('Feature: x_1')
title(['True Clusters  ,  ' Model]), 
axis([min(x(:,1))-1 max(x(:,1))+1 min(x(:,2))-1 max(x(:,2))+1])

end

