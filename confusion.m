function [ConfMatClass ConfMatAll Accuracy Error] = confusion(TrueLabels, FoundLabels)

% function [ConfMatClass ConfMatAll Accuracy Error] =
% eegc3_confusion_matrix(TrueLabels, FoundLabels)
%
% Function to recover a confusion matric of classification problem given
% the groiund truth classes and the classes found by a classifier
%
% Inputs:
%
% TrueLabels: Ground truth data labels
%
% FoundLabels: Labels predicted by a classifier
%
% Outputs: 
%
% ConfMatClass: Confusion Matrx ClassNum x ClassNum in percentages per
% class
%
% ConfMatAll: Confusion Matrx ClassNum x ClassNum in overall percentages
% 
% Accuracy: Total accuracy % (diagonal of Confusion Matrix)
%
% Error: Total Error % (100 - Accuracy)

if(~isvector(TrueLabels) || ~isvector(FoundLabels))
    disp('[eegc3_confusion_matrix] Labels must be vectors');
    ConfMat = [];
    Accuracy = [];
    Error = [];
    return;
end

N1 = length(TrueLabels);
N2 = length(FoundLabels);

if(N1 ~= N2)
    disp('[confusion] True and predicted labels must be the same size');
    ConfMat = [];
    Accuracy = [];
    Error = [];
    return;
else
    NSample = N1;
end

% Find number of classes
Classes = sort(unique(TrueLabels));
NClass = length(Classes);

% if(~isequal(Classes,[1:NClass])) 
%     NClass=Classes;
% end

if(~isequal(Classes,[1:NClass]) && ~isequal(Classes,[1:NClass]'))
    disp('[confusion] Classes must be a vector [1:N] or [1:N]');
end

% Compute Confusion Matrix

ConfMat = zeros(NClass);
ConfMatAll = zeros(NClass);
for i=1:NSample
    ConfMat(TrueLabels(i),FoundLabels(i)) = ...
        ConfMat(TrueLabels(i),FoundLabels(i)) + 1;
end

ConfMatAll = 100*ConfMat/sum(ConfMat(:));

ConfMatClass = zeros(NClass);
for i=1:NClass
    ConfMatClass(i,:) = 100*ConfMat(i,:)/sum(ConfMat(i,:));
end

Accuracy = trace(ConfMatAll);

Error = 100 - Accuracy;





