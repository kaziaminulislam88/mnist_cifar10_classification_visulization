clear all;close all;clc
%addpath(genpath('C:\Users\kisla001\Documents\MATLAB\SVM\libLinearSVM'));
addpath(genpath('C:\Users\kisla001\Documents\MATLAB\SVM\libSVM\libsvm-3.20\matlab'))
addpath(genpath('G:\001_Data\Kazi\nasa_Aug15\cnn_keras\transfer learning\cifar10\libsvm-3.22'))

%load('cifar10_resnet152_X_test_Nov4.mat')
%load('cifar10_resnet152_X_validation_Nov4.mat')
load('cifar10_resnet152_Y_test1_Nov4.mat')
load('cifar10_resnet152_y_validation1_Nov4.mat')

y_test=double(Y_test1);
y_validation=double(y_validation1);
%
intermediate_train=X_validation;
intermediate_test=X_test;

intermediate_train=reshape(intermediate_train,size(intermediate_train,1),size(intermediate_train,2)*size(intermediate_train,3)*size(intermediate_train,4));
intermediate_test=reshape(intermediate_test,size(intermediate_test,1),size(intermediate_test,2)*size(intermediate_test,3)*size(intermediate_test,4));
x_validation=double(intermediate_train);
x_test=double(intermediate_test);


disp('training accuracy for original data and label')
train_acc = svmtrain(y_validation, x_validation, '-q -v 5 -s 0 -t 3 -c 2.00000000');


disp('testing accuracy for original data and label')
%testing accuracy
modelSVM = svmtrain(y_validation, x_validation, '-q -s 0 -t 3 -c 2.00000000');
[predicted_label, testAcc, prob_estimates] = svmpredict(y_test, x_test, modelSVM); 
%
clearvars -except y_validation y_test

%flatten_2
load('cifar10_resnet152_finetuned_intermediate_predict_validation_Nov4.mat')
load('cifar10_resnet152_finetuned_intermediate_predict_test_Nov4.mat')

%intermediate_train=reshape(intermediate_train,size(intermediate_train,1),size(intermediate_train,2)*size(intermediate_train,3)*size(intermediate_train,4));
%intermediate_test=reshape(intermediate_test,size(intermediate_test,1),size(intermediate_test,2)*size(intermediate_test,3)*size(intermediate_test,4));
x_validation_flatten_2=double(intermediate_predict_validation);
x_test_flatten_2=double(intermediate_predict_test);

disp('training accuracy for flatten_2')
train_acc = svmtrain(y_validation, x_validation_flatten_2, '-q -v 5 -s 0 -t 3 -c 2.00000000');


disp('testing accuracy for flatten_2')
%testing accuracy
modelSVM = svmtrain(y_validation, x_validation_flatten_2, '-q -s 0 -t 3 -c 2.00000000');
[predicted_label, testAcc, prob_estimates] = svmpredict(y_test, x_test_flatten_2, modelSVM); 
%
clearvars -except y_validation y_test


%bn5c_branch2a
load('cifar10_resnet152_finetuned_intermediate_predict_validation_bn5c_branch2a_Nov4.mat')
load('cifar10_resnet152_finetuned_intermediate_predict_test_bn5c_branch2a_Nov4.mat')

intermediate_train=intermediate_predict_validation;
intermediate_test=intermediate_predict_test;
intermediate_train=reshape(intermediate_train,size(intermediate_train,1),size(intermediate_train,2)*size(intermediate_train,3)*size(intermediate_train,4));
intermediate_test=reshape(intermediate_test,size(intermediate_test,1),size(intermediate_test,2)*size(intermediate_test,3)*size(intermediate_test,4));
x_validation_bn5c_branch2a=double(intermediate_train);
x_test_bn5c_branch2a=double(intermediate_test);

disp('training accuracy for bn5c_branch2a')
train_acc = svmtrain(y_validation, x_validation_bn5c_branch2a, '-q -v 5 -s 0 -t 3 -c 2.00000000');


disp('testing accuracy for bn5c_branch2a')
%testing accuracy
modelSVM = svmtrain(y_validation, x_validation_bn5c_branch2a, '-q -s 0 -t 3 -c 2.00000000');
[predicted_label, testAcc, prob_estimates] = svmpredict(y_test, x_test_bn5c_branch2a, modelSVM); 

clearvars -except y_validation y_test

%res4b4_branch2a
load('cifar10_resnet152_finetuned_intermediate_predict_test_res4b4_branch2a_Nov4.mat')
load('cifar10_resnet152_finetuned_intermediate_predict_validation_res4b4_branch2a_Nov4.mat')

intermediate_train=intermediate_predict_validation;
intermediate_test=intermediate_predict_test;

intermediate_train=reshape(intermediate_train,size(intermediate_train,1),size(intermediate_train,2)*size(intermediate_train,3)*size(intermediate_train,4));
intermediate_test=reshape(intermediate_test,size(intermediate_test,1),size(intermediate_test,2)*size(intermediate_test,3)*size(intermediate_test,4));
x_validation_res4b4_branch2a=double(intermediate_train);
x_test_res4b4_branch2a=double(intermediate_test);

disp('training accuracy for res4b4_branch2a')
train_acc = svmtrain(y_validation, x_validation_res4b4_branch2a, '-q -v 5 -s 0 -t 3 -c 2.00000000');


disp('testing accuracy for res4b4_branch2a')
%testing accuracy
modelSVM = svmtrain(y_validation, x_validation_res4b4_branch2a, '-q -s 0 -t 3 -c 2.00000000');
[predicted_label, testAcc, prob_estimates] = svmpredict(y_test, x_test_res4b4_branch2a, modelSVM); 

clearvars -except y_validation y_test


%res3a_branch2a
load('cifar10_resnet152_finetuned_intermediate_predict_validation_res3a_branch2a_Nov4.mat')
load('cifar10_resnet152_finetuned_intermediate_predict_test_res3a_branch2a_Nov4.mat')

intermediate_train=intermediate_predict_validation;
intermediate_test=intermediate_predict_test;
intermediate_train=reshape(intermediate_train,size(intermediate_train,1),size(intermediate_train,2)*size(intermediate_train,3)*size(intermediate_train,4));
intermediate_test=reshape(intermediate_test,size(intermediate_test,1),size(intermediate_test,2)*size(intermediate_test,3)*size(intermediate_test,4));
x_validation_res3a_branch2a=double(intermediate_train);
x_test_res3a_branch2a=double(intermediate_test);

disp('training accuracy for res3a_branch2a')
train_acc = svmtrain(y_validation, x_validation_res3a_branch2a, '-q -v 5 -s 0 -t 3 -c 2.00000000');


disp('testing accuracy for res3a_branch2a')
%testing accuracy
modelSVM = svmtrain(y_validation, x_validation_res3a_branch2a, '-q -s 0 -t 3 -c 2.00000000');
[predicted_label, testAcc, prob_estimates] = svmpredict(y_test, x_test_res3a_branch2a, modelSVM); 

clearvars -except y_validation y_test

%res2a_branch2a
load('cifar10_resnet152_finetuned_intermediate_predict_test_res2a_branch2a_Nov4.mat')
load('cifar10_resnet152_finetuned_intermediate_predict_validation_res2a_branch2a_Nov4.mat')

intermediate_train=intermediate_predict_validation;
intermediate_test=intermediate_predict_test;
intermediate_train=reshape(intermediate_train,size(intermediate_train,1),size(intermediate_train,2)*size(intermediate_train,3)*size(intermediate_train,4));
intermediate_test=reshape(intermediate_test,size(intermediate_test,1),size(intermediate_test,2)*size(intermediate_test,3)*size(intermediate_test,4));
x_validation_res2a_branch2a=double(intermediate_train);
x_test_res2a_branch2a=double(intermediate_test);

disp('training accuracy for res2a_branch2a')
train_acc = svmtrain(y_validation, x_validation_res2a_branch2a, '-q -v 5 -s 0 -t 3 -c 2.00000000');


disp('testing accuracy for res2a_branch2a')
%testing accuracy
modelSVM = svmtrain(y_validation, x_validation_res2a_branch2a, '-q -s 0 -t 3 -c 2.00000000');
[predicted_label, testAcc, prob_estimates] = svmpredict(y_test, x_test_res2a_branch2a, modelSVM); 

clearvars -except y_validation y_test


