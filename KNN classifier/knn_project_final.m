clc; clear all
%Data file containing all dataset
load 'AlldataK.mat';

%Data file containing 120 records of the dataset for the training
load 'data.mat';

%Data file containing 6 records of the dataset for the testing
load 'TestdataK.mat';

%load true class of 6 records
load 'Trueclass6.mat';

%Traning Step
%reducing the 310 features to 4 features by calling an implemnted reduction
%function
reducedAllF = reduceF(AllX); %reducing all 126 record
reducedF = reduceF(X); %reducing just 120 record

%Traning the reduced features using fitcknn built-in function
TM = fitcknn (reducedF,Y); %traning the 120 record
TMA = fitcknn (reducedAllF,AllY); %traning the 126 record

%Editing the no. of nearest neighbours to give best accuracy.
TM.NumNeighbors = 11; %for 120 record
TMA.NumNeighbors = 11; %for all records

%Calculating the Crossval 
PMK = crossval(TM,'Kfold',10); %120
PMH = crossval(TM,'Holdout',0.1); %120
PMKA = crossval(TMA,'Kfold',10); %126
PMHA = crossval(TMA,'Holdout',0.1); %126

%Calculating error
errorK = PMK.kfoldLoss; %120
testdata=reducedAllF(PMK.Partition.test(1)); %not yet used
errorH = PMH.kfoldLoss; %120
errorKA = PMKA.kfoldLoss; %126
errorHA = PMHA.kfoldLoss; %126
accuracyK = 1 - errorK; %120
accuracyH = 1 - errorH; %120
accuracyKA = 1 - errorKA; %126
accuracyHA = 1 - errorHA; %126

%Calculatin '%' accuracy
accuracy_percentageK=accuracyK*100; %120
accuracy_percentageH=accuracyH*100; %120
accuracy_percentageKA=accuracyKA*100; %126
accuracy_percentageHA=accuracyHA*100; %126

%Displaying the accuraies from crossvalidation
disp('Training accuracy for 120 record using 10 Kfolds is');
disp(accuracy_percentageK);
disp('Training accuracy for 120 record using Holdout 0.1 is');
disp(accuracy_percentageH);
disp('Training accuracy for 126 record using 10 Kfolds is');
disp(accuracy_percentageKA);
disp('Training accuracy for 126 record using Holdout o.1 is');
disp(accuracy_percentageHA);

%tesing on the last 6 records
%reducing the fratures
reducedFtest = reduceF(Testdata);

%tesing using 120 records trained matrix
M=TM.predict(reducedFtest);
A=mean(M==True6);
disp('Testing accuracy for 120 record is');
disp(A*100);