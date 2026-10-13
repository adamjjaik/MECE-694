%% Reference
% This Code was adopted from: 
% Mostapha Kalami Heris, Yarpiz Evolutionary Algorithms Toolbox for MATLAB (YPEA), Yarpiz, 2020.
% Link: https://www.mathworks.com/matlabcentral/fileexchange/71373-ypea?s_tid=prof_contriblnk

% This code was modifies for "MECE 788: Applied Computational Intelligence
% for Engineering" at the Univerisyt of Alberta, Winter 2021.


function [NewTour] = CreateNeighbour(OldTour)
% Create a Neighbouring Solution Using Swap, Reversion, and Insertion

switch randi(3,1)
    case 1 % Swap
        NewTour = ApplySwap(OldTour);
    case 2 % Reversion
        NewTour = ApplyReversion(OldTour);
    case 3 % Insertion
        NewTour = ApplyInsertion(OldTour);   
end
end


function [NewTour] = ApplySwap(OldTour)
% Swap Operator for Permutation Problems

I = randsample(length(OldTour),2);
i1 = I(1); i2 = I(2);
NewTour = OldTour;
NewTour([i1 i2]) = OldTour([i2 i1]);
end

function [NewTour] = ApplyReversion(OldTour)
% Reversion Operator for Permutation Problems

I = randsample(length(OldTour),2);
i1 = min(I); i2 = max(I);
NewTour = OldTour;
NewTour(i1:i2) = OldTour(i2:-1:i1);
end

function [NewTour] = ApplyInsertion(OldTour)
% Insertion Operator for Permutation Problems

I = randsample(length(OldTour),2);
i1 = I(1); i2 = I(2);
if i1 < i2
    NewTour = OldTour([1:i1-1 i1+1:i2 i1 i2+1:end]);
else
    NewTour = OldTour([1:i2 i1 i2+1:i1-1 i1+1:end]);
end
end



