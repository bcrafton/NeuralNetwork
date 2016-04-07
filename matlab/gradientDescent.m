function [nn_params] = gradientDescent(f, nn_params)
    parpool(10);
    alpha = .8;
    num_iters = 100;
    tic;
    for iter = 1:num_iters
        [cost, grad] = f(nn_params);
        grad = grad * alpha;
        nn_params = nn_params - grad;
    end
    time_taken = toc;
    disp(time_taken);
    delete(gcp);
end
