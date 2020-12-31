function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

thetaMod=[0;theta(2:end)];
%thetaMod=thetaMod(:);

%J=(sum((X*theta-y).^2,"all")+lambda*sum(theta(2:end).^2,"all"))/2/m;
J=(sum((X*theta-y).^2,"all")+lambda*sum(thetaMod.^2,"all"))/2/m;

% X 12x2;  theta 2x1;  y 12x1

% =========================================================================

grad = grad(:);

%grad=1/m*sum((X*theta-y).*X,1);
%grad=grad + lambda/m*([0 theta(2:end)']);

grad=1/m*sum((X*theta-y).*X,1)'+lambda/m*thetaMod;

end
