%% Reference
% This Code was adopted from: 
% Mostapha Kalami Heris, Yarpiz Evolutionary Algorithms Toolbox for MATLAB (YPEA), Yarpiz, 2020.
% Link: https://www.mathworks.com/matlabcentral/fileexchange/71373-ypea?s_tid=prof_contriblnk

% This code was modifies for "MECE 788: Applied Computational Intelligence
% for Engineering" at the Univerisyt of Alberta, Winter 2021.

function [Model] = CreateModel(Problem, NumCities)
% This function creates a model for TSP 


switch Problem
    case 'Model 1' % 10 cities
        x = [49    48     5     8    24    47    72    30    95    96];
        y = [12    36    30    99    53    71     1    97    90    58];
        
    case 'Model 2' % 30 cities
        x = [44 40 5 36 96 50 77 17 98 87 92 50 52 84 61 45 71 32 21 9 96 66 43 75 43 37 61 66 52 23];
        y = [50 47 63 43 38 70 22 82 91 60 79 43 74 29 41 55 49 79 15 90 84 71 5 43 12 67 68 4 54 92];
        
    case 'Model 3' % Random with NumCities
        x = randi(100, 1, NumCities);
        y = randi(100, 1, NumCities);
end

% Calculate distance between each two cities
n = length(x);
D = zeros(n,n);
for i = 1:n-1
    for j = i+1:n
        D(i,j) = sqrt(sum(([x(i) y(i)] - [x(j) y(j)]) .^ 2));
        D(j,i) = D(i,j);
    end
end

% All coordinates are between 1 and 100
xmin = 1; xmax = 100;
ymin = 1; ymax=100;

% Record all info in structure Model
Model.n = n;
Model.x = x;
Model.y = y;
Model.D = D;
Model.xmin = xmin;
Model.xmax = xmax;
Model.ymin = ymin;
Model.ymax = ymax;
end