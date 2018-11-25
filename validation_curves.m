function [error_train, error_valid] = validation_curves(nn_params, n_inlayers, ...
                                                      n_hidden, n_outlayers, ...
                                                      X, y, Xvalid, ...
                                                      yvalid, lambdavec)

% Initialize vectors                                       
m = size(X,1);
error_train = zeros(numel(lambdavec),1);
error_valid = zeros(numel(lambdavec),1);
options     = optimset('MaxIter',100);

for i = 1:numel(lambdavec)
  
  temp = lambdavec(i);
  
  % Step 1 - Set cost function with different regularization parameter
  costfunction = @(t) nn_costfunction(t, n_inlayers, n_hidden, n_outlayers, ...
                                    X, y, temp);
   
  % Step 2 - Train neural network
  [nn_weights, cost] = fmincg(costfunction, nn_params, options);
  
  % Setp 3 - Evaluate error training and error validation without regularization (lambda = 0);
  error_train(i) = nn_costfunction(nn_weights, n_inlayers, n_hidden, n_outlayers, ...
                                   X((1:i),:), y(1:i), 0);
                                   
  error_valid(i) = nn_costfunction(nn_weights, n_inlayers, n_hidden, n_outlayers, ...
                                   Xvalid, yvalid, 0);
  
end


endfunction
