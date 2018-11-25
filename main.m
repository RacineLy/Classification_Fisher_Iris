% LEARNING ALGORITHM TO PREDICT IRIS FLOWER
%-------------------------------------------------------------------------------
% Problem type   :      Classification problem
% DataSet origin :      https://archive.ics.uci.edu/ml/datasets/iris
% Author         :      Racine LY, Ph.D 
% Date           :      11/24/2018
%-------------------------------------------------------------------------------

% Reset environment
clear; close all; clc;
load v;
rand("state",v);

% Read Data
fprintf('Loading Data ...\n');
Data = load('DataSet.txt');

% Identify and count classes
classes = unique(Data(:,end));

% Shuffle the data
fprintf('Shuffling Data ... \n');
newind = randperm(length([1:size(Data,1)]));
Mat    = zeros(size(Data));
for i = 1:size(Data,1)
  temp     = newind(i);
  Mat(i,:) = Data(temp,:);
end

% Define Train, Validation and Test Set rate
fprintf('Defining train, validation and test set ... \n');
nb_train = 0.6*size(Mat,1);
nb_valid = 0.2*size(Mat,1); 

% Training set
Xtrain = Mat(1:nb_train,1:(size(Data,2)-1));
ytrain = Mat(1:nb_train,end);
[m n]  = size(Xtrain);

% Validation set
Xvalid = Mat((nb_train+1):(nb_train+nb_valid),1:(size(Data,2)-1));
yvalid = Mat((nb_train+1):(nb_train+nb_valid),end);

% Test set
Xtest  = Mat((nb_train+nb_valid+1):end,1:(size(Data,2)-1));
ytest  = Mat((nb_train+nb_valid+1):end,end);

% Neural network parameters
n_inlayers  = n;
n_hidden    = 4;
n_outlayers = numel(classes);

% Initialize parameters
fprintf('Initialize Neural Network parameters ... \n');
epsilon1  = sqrt(6)/sqrt(n_inlayers + n_hidden);
epsilon2  = sqrt(6)/sqrt(n_hidden + n_outlayers);
theta1    = randomize_param(n_inlayers, n_hidden, epsilon1);
theta2    = randomize_param(n_hidden, n_outlayers, epsilon2);
nn_params = [theta1(:) ; theta2(:)]; 

% Test of cost and gradient function
fprintf('Testing Cost function and Backpropagation ... \n');
lambda      = 0;
[cost grad] = nn_costfunction(nn_params, n_inlayers, n_hidden, n_outlayers, ...
                              Xtrain, ytrain, lambda);
                             
% Check backpropagation with numerical approximation of gradient
fprintf('Checking gradient values with numerical approximation ... \n'); 
costfunction = @(t) nn_costfunction(t, n_inlayers, n_hidden, n_outlayers, ...
                                    Xtrain, ytrain, lambda);                                  
gradnum      = approxgrad(costfunction, nn_params);

% Learning curves
##[error_train, error_valid] = learning_curves(nn_params, n_inlayers, ...
##                                             n_hidden, n_outlayers, ...
##                                             Xtrain, ytrain, Xvalid, ...
##                                             yvalid, lambda);
                                             
% Validation curves
fprintf('Validation curve, choosing the right regularization parameter ... \n');
lambdavec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
[error_train, error_valid] = validation_curves(nn_params, n_inlayers, ...
                                             n_hidden, n_outlayers, ...
                                             Xtrain, ytrain, Xvalid, ...
                                             yvalid, lambdavec);
                                             
% Choose the lambda which minimize error_valid
u      = find(error_valid == min(error_valid));
lambda = lambdavec(u); 

% Training neural network
fprintf('Training neural network ... \n'); 
options                  = optimset('MaxIter',100);
[nn_weights, cost_train] = fmincg(costfunction, nn_params, options);

% Test traines neural network on Test Set
theta1nn = reshape(nn_weights(1:(n_hidden*(n_inlayers+1))), n_hidden, (n_inlayers+1));
theta2nn = reshape(nn_weights(((n_hidden*(n_inlayers+1))+1):end), n_outlayers, (n_hidden+1)); 
pred     = predict(theta1nn, theta2nn, Xtest);
Acc_fit  = mean(double(pred == ytest))*100;

% Plots
Plot_recipe;




