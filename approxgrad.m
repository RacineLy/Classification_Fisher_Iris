function gradnum = approxgrad(J, theta)
  
  perturb = zeros(size(theta));     % Initialize perturbation vector
  gradnum = zeros(size(theta));     % Initialize gradient approximation vector
  e       = 1e-4;
  
  for i = 1:numel(theta)
    
    perturb(i) = e;
    loss1      = J(theta - perturb);    % negative perturbation
    loss2      = J(theta + perturb);    % positive perturbation
    gradnum(i) = (loss2 - loss1)/(2*e); % numerical gradient approximation
    perturb(i) = 0;                     % Reset perturbation vector to zero
    
  end
    
endfunction
