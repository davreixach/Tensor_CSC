function [D,Dhat,results] = tensor_trainer(Dhat,Xhat,Yhat,para)
    
n1 = para.n1;
n2 = para.n2;
n3 = para.n3;
n4 = para.n4;
K = para.K;
N = para.N;
lambda = para.lambda;
max_iter = para.maxiter;
max_iter_x = para.maxiter_x;
max_iter_d = para.maxiter_d;
filter_szx = para.filter_szx;
filter_szy = para.filter_szy;

%% Optimization Parameters

error_obj = zeros(1,max_iter);
counter = 0;
error_obj_change_thresh = 1e-8;

%% Fixed point optimization
while(true)
    counter = counter + 1;
    if(counter > max_iter)
        break;
    end

    %Fixing the dictionary and updating the sparse codes
    fprintf('-----> Updateing X (Sparse Code) \n');
    [X,~,error_reg] = sparse_code_update_ADMM_2D(Dhat,Xhat,Yhat,n3,n4,K,N,lambda,max_iter_x);

    loss1 = lambda*sum(abs(X(:)));
    X_per = permute(X,[3, 4, 1, 2]);
    Xhat_per = fft2(X_per);
    Xhat = permute(Xhat_per,[3, 4, 1, 2]);

    if para.solveDict
        %Fixing the sparse code and updating the dictionary
        fprintf('-----> Updateing D (Dictionary Learning) \n');
        [D,~,error_reg_D] = dictionary_update_ADMM_2D(Dhat,Xhat,Yhat,n1,n3,n4,K,N,filter_szx,filter_szy,max_iter_d);

        loss2 = error_reg_D(end);
        D_per = permute(D,[3, 4, 1, 2]);
        Dhat_per = fft2(D_per);
        Dhat = permute(Dhat_per,[3, 4, 1, 2]);
    else
        D = nan;
        Dhat = nan;
        loss2 = 0;
    end

    error_obj(counter) = loss1 + loss2;
     if (counter == 1)
        error_obj_change = 0 ;
    else
        error_obj_change = norm(error_obj(end) - error_obj(end-1))/norm(error_obj(end-1));
     end
    
    fprintf('----------------------------------------------------------------------------- \n');
    fprintf('+ Iter: %1.0f  RegError: %1.6f \n',counter,error_obj(counter));
    fprintf('----------------------------------------------------------------------------- \n');

    if(counter > 2)
        if(error_obj_change < error_obj_change_thresh)
            break;
        end
    end
end

n_total = numel(X);
results.NNZ = nnz(X);
results.CR = n_total/results.NNZ;
results.PSNR = 20*log10(sqrt(n_total))-20*log10(error_reg); % assuming error_reg equals l2

end