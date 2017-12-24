fprintf('Loading and Visualizing Data ...\n')

load('ex3data1.mat'); % training data stored in arrays X, y
lambda = 0.1;
% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
theta = zeros(num_labels, n + 1);
fprintf('Size theta: ')
size(theta)

% Add ones to the X data matrix
X = [ones(m, 1) X];
fprintf('Size X: ')
size(X)
% Initialize some useful values
m = length(y); % number of training examples
fprintf('Size y: ')
size(y)
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

h=sigmoid(X*theta');
fprintf('Size h: ')
size(h)
y(1,1);
h(1,:);
y(1,1)*h(1,:);
sum(y(1,1)*h(1,:));
q=y.*h;
fprintf('Size q: ');
size(q);
sum(q,2);
size(sum(q,2))
a = [1,2,3;4,5,6];
b = [2,4,2];
a*b.';
c= [1,2;3,4;5,6];
d=[2;4;2];
%c.*d.'

firstpartofJ=sum((-y).*h,2);
secondpartofJ=sum((1-y).*h,2);
size(firstpartofJ);
size(secondpartofJ);
Jtemp=1/m*sum(firstpartofJ-secondpartofJ);
size(Jtemp);
Jtemp;
size(theta);


thetassquared=theta.*theta;
thetassquared(:,1)=0;
size(sum(sum(thetassquared,2)))
vv=lambda/(2*m)*(sum(sum(thetassquared,2)));
size(vv)
Jtemp2=Jtemp+vv;
fprintf('Size y: ');
size(y);
size(h-y);
size(sum(h-y,2))
tempsum=sum(h-y,2);
size(X);
size((1/m)*(X'*tempsum));
total=(1/m)*(X'*tempsum);

%tt=-(y'.*log(h))-((1-y').*log(1-h));
%uu=1/m*sum(tt);
%vv=lambda/(2*m)*(sum(theta.*theta)-theta(1,1)*theta(1,1));
%J=uu+vv;

%for j=1:size(X,2) 
%    grad(j,1)=1/m*sum((h'-y).*X(:,j));
%end

%for j=1:size(X,2) 
%    if j==1 
%        grad(j,1)=1/m*sum((h'-y).*X(:,j));
%    else
%        grad(j,1)=(1/m*sum((h'-y).*X(:,j)))+lambda/m*theta(j,1);
%end


grad = zeros(size(theta));
fprintf('Size grad: ');
size(grad)
fprintf('Size X'': ');
size(X')

tempsum=sum(h-y,2);
fprintf('Size tempsum: ');
size(tempsum)
fprintf('Size theta: ');
size(theta)
grad=(1/m)*(X'*tempsum);
size(grad)
size(theta)
sumtheta=sum(theta,1);
sumtheta(1)=0;
sumtheta;
size(grad)
size(sumtheta')
grad(2:end,:)=grad(2:end,:)+lambda/m*sumtheta(2:end)'
