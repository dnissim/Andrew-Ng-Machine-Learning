function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = []; %zeros(num_iters, 1);
min_Tolerance = 10^-3;

    for iter = 1:num_iters
    
        % ====================== YOUR CODE HERE ======================
        % Instructions: Perform a single gradient step on the parameter vector
        %               theta. 
        %
        % Hint: While debugging, it can be useful to print out the values
        %       of the cost function (computeCostMulti) and gradient here.
        %
    
            theta=theta-alpha/m*X'*(X*theta-y);
    
        % ============================================================
    
        % Save the cost J in every iteration    
        J_history(iter) = computeCostMulti(X, y, theta);
    
        % Check if change in cost function between interations
        % has reduced more than the minimum tolerance.
        % If so then stop calculating.
        % Also only check after iteration is greater than 1
        
         if iter>1
             if J_history(iter-1)-J_history(iter)<=min_Tolerance
                return
             end
         end
    end

end
