function g = gradsigmoid(z)
  
  % Compute sigmoid output
  var = sigmoid(z);
  
  % Compute sigmoid gradient value
  g   = var.*(1 - var);
  
endfunction
