%% Initialization
clear ; close all; clc
%% Load Data
load 'data.mat';
%% cross validation
[train,test] = crossvalind('HoldOut',Y,0.2);
X_train=X(train,:);
Y_train=Y(train,:);
X_test=X(test,:);
Y_test=Y(test,:);
save('trainData.mat','X_train','Y_train');
save('testData.mat','X_test','Y_test');