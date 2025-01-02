function [xPts, wPts, nPts] = SigmaPoints_cholesky(x,P,w0)
%Symmetric sampling 2n+1 sigma points
n=length(x);
nPts=2*n+1;

%The sigma points are placed one after another in a matrix
xPts = zeros(n,2*n+1); %memory allocation

M=chol(P,'lower');%Cholesky factorization
scale=sqrt(n/(1-w0));
for i=1:n
    xPts(:,i)=x+scale*M(:,i);
    xPts(:,i+n)=x-scale*M(:,i); 
end

%We add the average point at the end
xPts(:,nPts)=x;

%Weights 
wPts=ones(1,nPts)*(1-w0)/(2*n);
wPts(nPts)=w0;
