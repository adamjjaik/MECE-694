%% Reference
% This Code was adopted from: 
% Mostapha Kalami Heris, Yarpiz Evolutionary Algorithms Toolbox for MATLAB (YPEA), Yarpiz, 2020.
% Link: https://www.mathworks.com/matlabcentral/fileexchange/71373-ypea?s_tid=prof_contriblnk

% This code was modifies for "MECE 788: Applied Computational Intelligence
% for Engineering" at the Univerisyt of Alberta, Winter 2021.


function [Length] = TourLength(Tour, Model)
% This fucntion calculates the tour length

% Close the tour by adding the first city to the end
Tour = [Tour Tour(1)];
Length = 0;
for i = 1:length(Tour)-1
    Length = Length + Model.D(Tour(i), Tour(i+1));
end

end