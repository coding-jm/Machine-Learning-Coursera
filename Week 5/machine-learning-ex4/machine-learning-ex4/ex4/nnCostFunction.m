function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

X = [ones(m, 1) X]; %5000 401

hiddenLayerOutput = sigmoid(X*Theta1'); %5000 25

inputToFinalLayer=[ones(m,1) hiddenLayerOutput]; % 5000 26

finalLayerOutput=sigmoid(inputToFinalLayer*Theta2'); %5000 10
yToJ=zeros(size(y,1),num_labels);
for i = 1:num_labels
    for j = 1:(size(y,1))
        if y(j)==i
            yToJ(j,i)=1;
        end
    end
end

sumElement = -yToJ.*log(finalLayerOutput)-(1-yToJ).*log(1-finalLayerOutput);
J = 1/m*sum(sum(sumElement));

%REGULARIZATION

Theta1squared = Theta1.*Theta1;
size(Theta1squared);
Theta1squared(:,1) = 0;
Theta2squared = Theta2.*Theta2;
size(Theta1squared);
Theta2squared(:,1) = 0;
J = J + lambda/(2*m)*(sum(sum(Theta2squared))+sum(sum(Theta1squared)));
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

error3 = finalLayerOutput - yToJ; %5000 10 
%Theta2 10 26
error2 = (Theta2'*error3').*[ones(size(X*Theta1',1),1) sigmoidGradient(X*Theta1')]';
size(error2); %26 5000
error2 = error2(2:end,:);

D1 = error2*X; %25 401
D2 = error3'*inputToFinalLayer;
Theta1_grad = Theta1_grad + 1/m*D1;
Theta2_grad = Theta2_grad + 1/m*D2;
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m*Theta2(:,2:end);

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end