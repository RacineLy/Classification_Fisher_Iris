function w = randomize_param(Lin, Lout, epsilon)
  
  % Compute randomized parameters values
  w = rand(Lout, (Lin+1))*(2*epsilon) - epsilon;  
  
endfunction
