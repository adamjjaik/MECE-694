function [Map, C] = MatchClusters(yTrue, Cluster, K)
% MatchClusters  Match k-means cluster IDs to true class labels.
%   k-means numbers its clusters arbitrarily (cluster 3 might be "sitting"),
%   so before computing accuracy each cluster must be paired with one class.
%   [Map, C] = MatchClusters(yTrue, Cluster, K) finds the one-to-one pairing
%   that maximizes the number of correctly matched samples (Hungarian
%   algorithm, via matchpairs).
%   Map(c) = class assigned to cluster c
%   C      = K x K contingency table, C(i,j) = # samples of class i in cluster j
C = confusionmat(yTrue(:), Cluster(:), 'Order', 1:K);   % rows = classes, columns = clusters
Pairs = matchpairs(C, 0, 'max');                        % [class, cluster] pairs
Map = zeros(K, 1);
Map(Pairs(:,2)) = Pairs(:,1);
end
