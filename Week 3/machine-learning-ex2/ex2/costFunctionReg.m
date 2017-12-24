function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



h=sigmoid(theta'*X');
for j=1:size(X,2) 
    if j==1 
        grad(j,1)=1/m*sum((h'-y).*X(:,j));
    else
        grad(j,1)=(1/m*sum((h'-y).*X(:,j)))+lambda/m*theta(j,1);
end

tt=-(y'.*log(h))-((1-y').*log(1-h));
uu=1/m*sum(tt);
vv=lambda/(2*m)*(sum(theta.*theta)-theta(1,1)*theta(1,1));
J=uu+vv;


% =============================================================

end
