%% Initialization
clear ; close all; clc
%% Load Data
load 'Alldata.mat';
%% Features reduction
Xdata=reduceF(X);
Ydata=Y;
val=3; %%folds value
fold=crossvalind('kFold',Y,val);
Acc_train=0;
Acc_test=0;
SN_train=0;
Sp_train=0;
SN_test=0;
Sp_test=0;
%fprintf('folds:\n');
%fprintf('      %f\n',fold); 
for f=1:val,
    %%get fold
    X=Xdata(fold~=f,:);
    X_test=Xdata(fold==f,:);
    Y=Ydata(fold~=f,:);
    Y_test=Ydata(fold==f,:);
    %fprintf('fold no %f\n',f); 
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
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
fprintf('Accuracy: %f\n',mean(yp==Y)); 
Acc_train=Acc_train+mean(yp==Y);
c=confusionmat(Y,yp,'order',[2;1]); %confusion matrix it same like it should be transposed
s=sum(c,2);         
    % TP+FN
    % TN+FP
% Sensitivity= TP/(TP+FN)
fprintf('Sensitivity: %f\n',c(1,1)/s(1,1)); 
SN_train=SN_train+c(1,1)/s(1,1);
% Specificity= TN/(TN+FP)
fprintf('Specificity: %f\n',c(2,2)/s(2,1)); 
Sp_train=Sp_train+c(2,2)/s(2,1);
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
Acc_test = Acc_test + mean(yp_test==Y_test);
c=confusionmat(Y_test,yp_test,'order',[2;1]); %confusion matrix it same like it should be transposed
s=sum(c,2);         
    % TP+FN
    % TN+FP
% Sensitivity= TP/(TP+FN)
fprintf('Test Sensitivity: %f\n',c(1,1)/s(1,1)); 
SN_test=SN_test+c(1,1)/s(1,1);
% Specificity= TN/(TN+FP)
fprintf('Test Specificity: %f\n',c(2,2)/s(2,1)); 
Sp_test=Sp_test+c(2,2)/s(2,1);
% fprintf('================perdiction values================ \n');
% fprintf('  %f\n',yp_test);
fprintf('================================================ \n');
end;
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
fprintf('================Avg results================ \n');
fprintf('================Train================ \n');
fprintf('Train Accuracy:%f\nTrain Sensitivity:%f\nTrain Specificity:%f\n',...
    Acc_train/val*100,SN_train/val*100,Sp_train/val*100);
fprintf('================Test================ \n');
fprintf('Test Accuracy:%f\n Test Sensitivity:%f\nTest Specificity:%f\n',...
    Acc_test/val*100,SN_test/val*100,Sp_test/val*100);
fprintf('================That''s All================ \n');
