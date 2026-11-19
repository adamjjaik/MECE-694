function [ObjValue, AddInfo] = ClusteringObjValue(c, x, Method)
% This function computes the objecitve value for clustering for input data
% "x" and cluster centers "c"

% Clustering Objecitve is: sum of distance of all data points from their
% closest cluster center

% Extract the location of cluster centers
C = c(:,1:end-1);

% Extract the activation of cluster centers
CC_activation = c(:,end);

% We Make sure that at least two cluster centers are activated
if sum(CC_activation >= 0.5) < 2
    [~, SortOrder] = sort(CC_activation, 'descend');
    CC_activation(SortOrder(1:2)) = 1;
end

% Only keep clusters centers that are activated
C = C(CC_activation >= 0.5, :);

switch Method
    case 'WCD'
        [ObjValue, AddInfo] = WCD(C, x);
        
    case 'DB_Index'
        [ObjValue, AddInfo] = DB_Index(C, x);         
end
AddInfo.CC_activation = double(CC_activation >= 0.5);
end

function [ObjValue, AddInfo] = WCD(C, x)
% This function computes the "withing class distance" as objective function
% This Code was adopted from: 
% https://yarpiz.com/64/ypml101-evolutionary-clustering

% This code was modifies for "MECE 788: Applied Computational Intelligence
% for Engineering" at the Univerisyt of Alberta, Winter 2021.

% Calculate Pairwise distance between each data point and cluster center
DistanceMetric = 'euclidean'; % Options are: euclidean, mahalanobis, cityblock,
% minkowski, chebychev, cosine, correlation, etc
d = pdist2(x, C, DistanceMetric);

% Find closest cluseter center to each data point, then assign data point
% to that cluster
[Min_d, Ind_d] = min(d, [], 2);

% Sum of Within-Cluster Distances
ObjValue = sum(Min_d);

AddInfo.d = d;
AddInfo.Min_d = Min_d;
AddInfo.Ind_d = Ind_d;
AddInfo.ObjValue = ObjValue;
AddInfo.C = C;
end

function [ObjValue, AddInfo] = DB_Index(C, x)
% This function computes the DB Index which is defined as the ratio of
% "withing class distance" over "between class distance"

% This Code was adopted from: 
% https://yarpiz.com/64/ypml101-evolutionary-clustering

% This code was modifies for "MECE 788: Applied Computational Intelligence
% for Engineering" at the Univerisyt of Alberta, Winter 2021.

% Number of Active Clusters
NumCluster = size(C,1);

% Calculate Pairwise distance between each data point and cluster center
DistanceMetric = 'euclidean'; % Options are: euclidean, mahalanobis, cityblock,
% minkowski, chebychev, cosine, correlation, etc
d = pdist2(x, C, DistanceMetric);

% Find closest cluseter center to each data point, then assign data point
% to that cluster
[Min_d, Ind_d] = min(d, [], 2);

% Define Parameter of DB Index --> q
q = 2; % 2-norm

% Calculate Within Class Distance S
S = zeros(NumCluster, 1);
for i = 1:NumCluster % for data in each cluster
    if sum(Ind_d == i) > 0 % If there are some data points in that cluster
        S(i) = (mean(Min_d(Ind_d == i).^q))^(1/q);
    else % if the cluster is empty, give a large objective value
        S(i) = 10 * norm(max(x)-min(x));
    end
end

% Define Parameter of DB Index --> t
t = 2; % 2-norm

% Compute cluster center to cluster center distance
D = pdist2(C, C, 'minkowski', t);

% Calculate the ration of "S" over "D"
r = zeros(NumCluster);
for i = 1:NumCluster % for each cluster
    for j = i+1:NumCluster % assess it with other clusters
        r(i,j) = (S(i)+S(j))/D(i,j);
        r(j,i) = r(i,j);
    end
end

% Get the max of r and return it as R
R = max(r);

% Objective value is the mean of R
ObjValue = mean(R);

AddInfo.d = d;
AddInfo.Min_d = Min_d;
AddInfo.Ind_d = Ind_d;
AddInfo.DB = ObjValue;
AddInfo.S = S;
AddInfo.D = D;
AddInfo.r = r;
AddInfo.R = R;
AddInfo.C = C;
end
