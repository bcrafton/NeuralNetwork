function [nn_params] = gradientDescent(f, nn_params)
    matlabpool(30);
    alpha = .5;
    num_iters = 100;
    tic;
    for iter = 1:num_iters
        [~, grad] = f(nn_params);
        grad = grad * alpha;
        nn_params = nn_params - grad;
        if (mod(iter+1, 100) == 0)
            disp(toc);		
        end
    end
    matlabpool close;
end
