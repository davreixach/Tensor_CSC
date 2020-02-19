function [D,Dhat,X,Xhat,Y,Yhat,params_sizes] = load_data2(b,K)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

if nargin < 2
    K = 100;
end

I = b;  [n3,n4,n1,N] = size(I);     n2 = 1;

D = randn(n1,K,n3,n4);
norm_matrix = sqrt(sum(sum(sum( conj(D).*D , 3),4),1)) ;
ind = norm_matrix>1;
D(:,ind,:,:) = D(:,ind,:,:)./ repmat(norm_matrix(ind),n1,1,n3,n4);
D_per = permute(D,[3, 4, 1, 2]);
Dhat_per = fft2(D_per);
Dhat = permute(Dhat_per,[3, 4, 1, 2]);

X = zeros(K,N,n3,n4);
X_per = permute(X,[3, 4, 1, 2]);
Xhat_per = fft2(X_per);
Xhat = permute(Xhat_per,[3, 4, 1, 2]);

Y = permute(I,[3,4,1,2]);
norm_matrix = sqrt(sum(sum(sum( conj(Y).*Y , 3),4),1)) ;
ind = norm_matrix>1;
Y(:,ind,:,:) = Y(:,ind,:,:)./ repmat(norm_matrix(ind),n1,1,n3,n4);
Y_per = permute(Y,[3, 4, 1, 2]);
Yhat_per = fft2(Y_per);
Yhat = permute(Yhat_per,[3, 4, 1, 2]);

params_sizes = [n1, n2, n3, n4, N, K];

end
