function P = NeighborhoodPreservation(Xhigh, Z, k)
% NeighborhoodPreservation  Fraction of each point's k nearest neighbours in the
% original space that remain among its k nearest neighbours in the embedding Z.
% Returns the mean over all points (1 = local structure perfectly preserved).
% Labels are not used.
idxH = knnsearch(Xhigh, Xhigh, 'K', k + 1);  idxH = idxH(:, 2:end);   % drop self
idxL = knnsearch(Z,     Z,     'K', k + 1);  idxL = idxL(:, 2:end);
n = size(Xhigh, 1);
overlap = zeros(n, 1);
for i = 1:n
    overlap(i) = numel(intersect(idxH(i,:), idxL(i,:))) / k;
end
P = mean(overlap);
end