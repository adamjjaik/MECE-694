function [Offspring] = Mutation_Incomplete(Parent, mu)
% This Function Perform Bianry Mutation by Changing Some Genes from 0 to 1
% or Vice Versa.

Offspring = [];

% *********** WRITE YOUR CODE HERE *************
% How Many Genes Must Undergo Mutation (Calculate Using mu)
% NumGenes Must be between 1 and length(Parent)

% Select NumGenes of Parent Randomly

% Apply Mutation


% **********************************************

% To Make Sure That At Least One Feature Is Selected (Dont need to change
% the following)
if sum(Offspring) == 0
    Offspring = Parent;
end
end