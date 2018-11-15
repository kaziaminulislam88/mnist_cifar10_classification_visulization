clear all;close all;clc
%addpath(genpath('C:\Users\kisla001\Documents\MATLAB\SVM\libLinearSVM'));
addpath(genpath('C:\Users\kisla001\Documents\MATLAB\SVM\libSVM\libsvm-3.20\matlab'))
addpath(genpath('G:\001_Data\Kazi\nasa_Aug15\cnn_keras\transfer learning\cifar10\libsvm-3.22'))
load('cifar10_resnet_50_Y_train_org_oct26.mat')
load('G:\001_Data\Kazi\nasa_Aug15\cnn_keras\transfer learning\cifar10\cifar10_resnet_50_output_predict_train_oct26.mat')
load('G:\001_Data\Kazi\nasa_Aug15\cnn_keras\transfer learning\cifar10\cifar10_resnet_50_Y_train_oct26.mat')


y_train=double(Y_train_org);

x_train=double(output_predict_train);

disp('training accuracy')
train_acc = svmtrain(y_train, x_train, '-q -v 5 -s 0 -t 3 -c 2.00000000');

load('cifar10_resnet_50_output_predict_test_oct26.mat')
load('cifar10_resnet_50_y_test_org_oct26.mat')

x_test=double(output_predict_test);
y_test=double(y_test_org);

disp('testing accuracy')
%testing accuracy
modelSVM = svmtrain(y_train, x_train, '-q -s 0 -t 3 -c 2.00000000');
[predicted_label, testAcc, prob_estimates] = svmpredict(y_test, x_test, modelSVM); 

clearvars -except y_train y_test

%for resnet 152 dense 1000 layer
load('cifar10_resnet152_output_predict_train_Nov1.mat');
load('cifar10_resnet152_output_predict_test_Nov1.mat')
x_train_resnet152_dense1000=double(output_predict_train);
x_test_resnet152_dense1000=double(output_predict_test);

disp('training accuracy for resnet 152 dense 1000 layer')
train_acc = svmtrain(y_train, x_train_resnet152_dense1000, '-q -v 5 -s 0 -t 3 -c 2.00000000');

disp('testing accuracy for resnet 152 dense 1000 layer')
%testing accuracy
modelSVM = svmtrain(y_train, x_train_resnet152_dense1000, '-q -s 0 -t 3 -c 2.00000000');
[predicted_label, testAcc, prob_estimates] = svmpredict(y_test, x_test_resnet152_dense1000, modelSVM); 

clearvars -except y_train y_test

%flatten1 layer
load('flatten_1cifar10_resnet152_output_intermediate_train_Nov1.mat')
load('flatten_1cifar10_resnet152__intermediate_test_Nov1.mat')
x_train_resnet152_flatten1=double(intermediate_train);
x_test_resnet152_flatten1=double(intermediate_test);

clear intermediate_train intermediate_test

disp('training accuracy for resnet 152 flatten1 layer')
train_acc = svmtrain(y_train, x_train_resnet152_flatten1, '-q -v 5 -s 0 -t 3 -c 2.00000000');

disp('testing accuracy for resnet 152 flatten1 layer')
%testing accuracy
modelSVM = svmtrain(y_train, x_train_resnet152_flatten1, '-q -s 0 -t 3 -c 2.00000000');
[predicted_label, testAcc, prob_estimates] = svmpredict(y_test, x_test_resnet152_flatten1, modelSVM); 

clearvars -except y_train y_test

%bn5c_branch2a layer
load('bn5c_branch2acifar10_resnet152_output_intermediate_train_Nov1.mat')
 load('bn5c_branch2acifar10_resnet152__intermediate_test_Nov1.mat')

intermediate_train=reshape(intermediate_train,size(intermediate_train,1),size(intermediate_train,2)*size(intermediate_train,3)*size(intermediate_train,4));
intermediate_test=reshape(intermediate_test,size(intermediate_test,1),size(intermediate_test,2)*size(intermediate_test,3)*size(intermediate_test,4));
x_train_resnet152_bn5c=double(intermediate_train);
x_test_resnet152_bn5c=double(intermediate_test);

clear intermediate_train intermediate_test

disp('training accuracy for resnet 152 bn5c_branch2a layer')
train_acc = svmtrain(y_train, x_train_resnet152_bn5c, '-q -v 5 -s 0 -t 3 -c 2.00000000');


disp('testing accuracy for resnet 152 bn5c_branch2a layer')
%testing accuracy
modelSVM = svmtrain(y_train, x_train_resnet152_bn5c, '-q -s 0 -t 3 -c 2.00000000');
[predicted_label, testAcc, prob_estimates] = svmpredict(y_test, x_test_resnet152_bn5c, modelSVM); 

clearvars -except y_train y_test

%res4b4_branch2a
load('res4b4_branch2a_cifar10_resnet152_output_intermediate_train_Nov1.mat')
load('res4b4_branch2a_cifar10_resnet152__intermediate_test_Nov1.mat')


intermediate_train=reshape(intermediate_train,size(intermediate_train,1),size(intermediate_train,2)*size(intermediate_train,3)*size(intermediate_train,4));
intermediate_test=reshape(intermediate_test,size(intermediate_test,1),size(intermediate_test,2)*size(intermediate_test,3)*size(intermediate_test,4));
x_train_resnet152_res4b4_branch2a=double(intermediate_train);
x_test_resnet152_res4b4_branch2a=double(intermediate_test);

clear intermediate_train intermediate_test


disp('training accuracy for resnet 152 res4b4_branch2a layer')
train_acc = svmtrain(y_train, x_train_resnet152_res4b4_branch2a, '-q -v 5 -s 0 -t 3 -c 2.00000000');


disp('testing accuracy for resnet 152 res4b4_branch2a layer')
%testing accuracy
modelSVM = svmtrain(y_train, x_train_resnet152_res4b4_branch2a, '-q -s 0 -t 3 -c 2.00000000');
[predicted_label, testAcc, prob_estimates] = svmpredict(y_test, x_test_resnet152_res4b4_branch2a, modelSVM);

clearvars -except y_train y_test


%res3a_branch2a
load('res3a_branch2a_cifar10_resnet152_output_intermediate_train_Nov1.mat')
load('res3a_branch2a_cifar10_resnet152__intermediate_test_Nov1.mat')


intermediate_train=reshape(intermediate_train,size(intermediate_train,1),size(intermediate_train,2)*size(intermediate_train,3)*size(intermediate_train,4));
intermediate_test=reshape(intermediate_test,size(intermediate_test,1),size(intermediate_test,2)*size(intermediate_test,3)*size(intermediate_test,4));
x_train_resnet152_res3a_branch2a=double(intermediate_train);
x_test_resnet152_res3a_branch2a=double(intermediate_test);

clear intermediate_train intermediate_test


disp('training accuracy for resnet 152 res3a_branch2a layer')
train_acc = svmtrain(y_train, x_train_resnet152_res3a_branch2a, '-q -v 5 -s 0 -t 3 -c 2.00000000');


disp('testing accuracy for resnet 152 res3a_branch2a layer')
%testing accuracy
modelSVM = svmtrain(y_train, x_train_resnet152_res3a_branch2a, '-q -s 0 -t 3 -c 2.00000000');
[predicted_label, testAcc, prob_estimates] = svmpredict(y_test, x_test_resnet152_res3a_branch2a, modelSVM);

clear intermediate_train intermediate_test
clearvars -except y_train y_test

%res2a_branch2a

load('res2a_branch2a_cifar10_resnet152_output_intermediate_train_Nov1.mat')
load('res2a_branch2a_cifar10_resnet152__intermediate_test_Nov1.mat')


intermediate_train=reshape(intermediate_train,size(intermediate_train,1),size(intermediate_train,2)*size(intermediate_train,3)*size(intermediate_train,4));
intermediate_test=reshape(intermediate_test,size(intermediate_test,1),size(intermediate_test,2)*size(intermediate_test,3)*size(intermediate_test,4));
x_train_resnet152_res2a_branch2a=double(intermediate_train);
x_test_resnet152_res2a_branch2a=double(intermediate_test);

clear intermediate_train intermediate_test


disp('training accuracy for resnet 152 res2a_branch2a layer')
train_acc = svmtrain(y_train, x_train_resnet152_res2a_branch2a, '-q -v 5 -s 0 -t 3 -c 2.00000000');


disp('testing accuracy for resnet 152 res2a_branch2a layer')
%testing accuracy
modelSVM = svmtrain(y_train, x_train_resnet152_res2a_branch2a, '-q -s 0 -t 3 -c 2.00000000');
[predicted_label, testAcc, prob_estimates] = svmpredict(y_test, x_test_resnet152_res2a_branch2a, modelSVM);

clearvars -except y_train y_test
clear intermediate_train intermediate_test



