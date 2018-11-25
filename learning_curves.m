function [error_train, error_valid] = learning_curves(nn_params, n_inlayers, ...
                                                      n_hidden, n_outlayers, ...
                                                      X, y, Xvalid, ...
                                                      yvalid, lambda)

% Initialize vectors                                       
m = size(X,1);
error_train = zeros(m,1);
error_valid = zeros(m,1);
options     = optimset('MaxIter',50);

for i = 1:m
  
  % Step 1 - Set cost function with different training set size
  costfunction = @(t) nn_costfunction(t, n_inlayers, n_hidden, n_outlayers, ...
                                    X((1:i),:), y(1:i), lambda);
   
  % Step 2 - Train neural network
  [nn_weights, cost] = fmincg(costfunction, nn_params, options);
  
  % Setp 3 - Evaluate error training and error validation without regularization (lambda = 0);
  error_train(i) = nn_costfunction(nn_weights, n_inlayers, n_hidden, n_outlayers, ...
                                   X((1:i),:), y(1:i), 0);
                                   
  error_valid(i) = nn_costfunction(nn_weights, n_inlayers, n_hidden, n_outlayers, ...
                                   Xvalid, yvalid, 0);
                                  
  disp([(i/m)*100 error_train(i) error_valid(i)]);
  
end


endfunction
