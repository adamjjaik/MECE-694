%% Reference
% This Code was adopted from: 
% Mostapha Kalami Heris, Yarpiz Evolutionary Algorithms Toolbox for MATLAB (YPEA), Yarpiz, 2020.
% Link: https://www.mathworks.com/matlabcentral/fileexchange/71373-ypea?s_tid=prof_contriblnk

% This code was modifies for "MECE 788: Applied Computational Intelligence
% for Engineering" at the Univerisyt of Alberta, Winter 2021.

function [Index] = RouletteWheelSelection(P)
% This function performs Roulette Wheel Selection
% based on given probabilities P

i = rand; % Select a random number
C = cumsum(P); % Cumulative sum of P
Index = find(i <= C, 1, 'first');
end