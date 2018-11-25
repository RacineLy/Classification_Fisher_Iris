function [cost grad] = nn_costfunction(nn_params, n_inlayers, n_hidden, ...
                                      n_outlayers, X, y, lambda);          
              
% Useful size parameters
m = size(X,1);

% Define design matrix
X = [ones(m,1) X];

% Reshape neural network weights
theta1 = reshape(nn_params(1:(n_hidden*(n_inlayers+1))), n_hidden, (n_inlayers+1));
theta2 = reshape(nn_params(((n_hidden*(n_inlayers+1))+1):end), n_outlayers, (n_hidden+1));  

% Initialize gradient
theta1nn = zeros(size(theta1));
theta2nn = zeros(size(theta2));

% Perform Feedforward computations
a1 = X';
z2 = theta1*a1;
a2 = sigmoid(z2);
a2 = [ones(1,size(a2,2)); a2];
z3 = theta2*a2;
a3 = sigmoid(z3);
hk = a3;

% Vecetorize classes with binary value 0/1
yk = zeros(n_outlayers,m);
for i = 1:m
  yk(y(i),i) = 1;
end

% Compute Cost without regularization
hyp  = -yk.*log(hk) - (1-yk).*log(1-hk);
cost = (1/m)*sum(sum(hyp));

% Compute cost with regularization
temp1    = theta1(:,(2:end)).^2;
temp2    = theta2(:,(2:end)).^2;
reg_term = (lambda/(2*m))*(sum(sum(temp1)) + sum(sum(temp2)));
cost     = cost + reg_term; 

% Compute gradient --> Backpropagation
for i = 1:m
  
  % Step 1 - FeedForward to compute all activation functions
  a1 = X(i,:)';
  z2 = theta1*a1;
  a2 = sigmoid(z2);
  a2 = [1; a2];
  z3 = theta2*a2;
  a3 = sigmoid(z3);
  
  % Step 2 --> Compute error at output layer
  delta3 = a3 - ([1:n_outlayers] == y(i))';
  
  % Step 3 --> Backpropagate error for hidden layers
  delta2 = (theta2)'*delta3.*gradsigmoid([1;z2]);
  delta2 = delta2(2:end);
  
  % Step 4 --> Accumulate gradient
  theta1nn = theta1nn + delta2*(a1');
  theta2nn = theta2nn + delta3*(a2');
  
end

% Gradient with regularization
% No regularization for first column
theta1nn(:,1) = (1/m)*theta1nn(:,1);
theta2nn(:,1) = (1/m)*theta2nn(:,1);

% No regularization for first column
theta1nn(:,(2:end)) = (1/m)*theta1nn(:,(2:end)) + (lambda/m)*theta1nn(:,(2:end));
theta2nn(:,(2:end)) = (1/m)*theta2nn(:,(2:end)) + (lambda/m)*theta2nn(:,(2:end));

grad = [theta1nn(:) ; theta2nn(:)];
            
              
endfunction
