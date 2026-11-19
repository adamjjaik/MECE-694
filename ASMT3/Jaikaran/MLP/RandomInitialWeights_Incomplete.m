function [Initial_theta] = RandomInitialWeights_Incomplete(In, Out)
% This function generates random initial weights for an MLP layer with
% "In" incoming connections and "Out" outgoing connections

% Initial_theta: Matrix of size(Out, 1 + In), 

% Write a Code to Calculate "Initial_theta" 
Initial_theta = zeros(Out, 1 + In);

% ********************* Write You Code Here *********************
% Note 1: use "rand" function and parameter "Range" to create random
% weights in the range of -Range to +Range
% Note 2: first column of "Initial_theta" contains the "bias" terms

Range = 0.1; % Don't Change This Line
Initial_theta = rand(Out, 1 + In) * 2 * Range - Range;

% *****************************************************************

end