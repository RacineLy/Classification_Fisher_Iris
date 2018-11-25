function p = predict(theta1nn, theta2nn, X)
  
  m  = size(X,1);
  h1 = sigmoid([ones(m,1) X]*theta1nn');
  h2 = sigmoid([ones(m,1) h1]*theta2nn');
  
  [dummy p] = max(h2, [], 2);
  
endfunction
