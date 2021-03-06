%% Initialization
clear ; close all; clc
%% cross validation Hold out
%%crossvalidation(0.5)
%% Load Data
load 'trainData.mat';
%% Features reduction
X=reduceF(X_train);
Y=Y_train;
%% Features'mean and std
Fmu=mean(X);
Fsigma=std(X);
%% getting features z values
ZF=getZ(X,Fmu,Fsigma);
%% classes probability
p1=mean(Y==1);           %probability of Binary class 1=acceptable
p2=mean(Y==2);           %probability of Binary class 2=unacceptable

%% class 1=acceptable
mu1=mean(X(Y==1,:));
sigma1=std(X(Y==1,:));

%% class 2=unacceptable
mu2=mean(X(Y==2,:));
sigma2=std(X(Y==2,:));

%% calculating accuracy
% conditional normal probability of X given Binary class 1=acceptable
XY1=condnorm(X,mu1,sigma1)*p1; %% may be we switch X with ZF
% conditional normal probability of X given Binary class 2=unacceptable
XY2=condnorm(X,mu2,sigma2)*p2; %% may be we switch X with ZF

% prediction y given X features
yp=zeros(size(Y)); % prediction of Y
for i=1:length(yp),
    if XY1(i,1)>XY2(i,1),
        yp(i,1)=1;
    else
         yp(i,1)=2;
    end;
end;
fprintf('Accuracy: %f\n',mean(yp==Y)); 
c=confusionmat(Y,yp); %confusion matrix it same like it should be transposed
s=sum(c,2);         
    % TP+FN
    % TN+FP
% Sensitivity= TP/(TP+FN)
fprintf('Sensitivity: %f\n',c(2,2)/s(2,1)); 
% Specificity= TN/(TN+FP)
fprintf('Specificity: %f\n',c(1,1)/s(1,1)); 
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
fprintf('===================perdiction=================== \n');
%% prediction new data
load 'testData.mat';
X_test=reduceF(X_test);
% conditional normal probability of X given Binary class 1=acceptable
XYtest1=condnorm(X_test,mu1,sigma1)*p1; 
% conditional normal probability of X given Binary class 2=unacceptable
XYtest2=condnorm(X_test,mu2,sigma2)*p2;

% prediction y given X features
yp_test=zeros(size(Y_test)); % prediction of Y
for i=1:length(yp_test),
    if XYtest1(i,1)>XYtest2(i,1),
        yp_test(i,1)=1;
    else
         yp_test(i,1)=2;
    end;
end;
fprintf('Test Accuracy: %f\n',mean(yp_test==Y_test)); 
c=confusionmat(Y_test,yp_test); %confusion matrix it same like it should be transposed
s=sum(c,2);         
    % TP+FN
    % TN+FP
% Sensitivity= TP/(TP+FN)
fprintf('Test Sensitivity: %f\n',c(2,2)/s(2,1)); 
% Specificity= TN/(TN+FP)
fprintf('Test Specificity: %f\n',c(1,1)/s(1,1)); 
fprintf('================perdiction values================ \n');
 fprintf('  %f\n',yp_test);
fprintf('================That''s All================ \n');