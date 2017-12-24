function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
initial_theta = zeros(n+1, 1);
theta=initial_theta;
X*theta;
%size(theta) %401 1
%size(X); %401 5000
%size(X*theta) %5000 1
%size(h); %5000 1
%size(log(h))
%size(y); %5000 1
h=sigmoid(X*theta);
%size(log(h)); %10 5000

firstpartofJ=-y.*log(h);

secondpartofJ=(1-y).*log(1-h);
%size(firstpartofJ); 5000 1

%size(secondpartofJ); 5000 1
Jtemp=1/m*sum(firstpartofJ-secondpartofJ);
%size(Jtemp);


thetassquared=theta.*theta;
%size(theta);
%size(thetassquared);
%size(theta.*theta);
thetassquared(1,:)=0;

sum(thetassquared);
%size(sum(sum(thetassquared,2))); %[1 1]
vv=(lambda/(2*m))*sum(thetassquared);
%size(vv); % 1 1 
J=Jtemp+vv;

% =============================================================

%grad = grad(:);

%size(X'); %401 5000
%size(X); %5000 401
%size(h-y) %10 5000
%size(theta) %10 401
%%size(h) %5000 1
%size(y) %5000 1
sumelement=X'*(h-y);
%size(sumelement) %401 1

grad=(1/m)*(sumelement);
%size(grad);

grad(2:end,:)=grad(2:end,:)+lambda/m*theta(2:end);
size(grad);

end
