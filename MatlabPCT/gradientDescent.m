function [nn_params] = gradientDescent(f, nn_params, X, y, input_layer_size, hidden_layer_size, num_labels)
    matlabpool(4);
    alpha = .8;
    num_iters = 1000;
    tic;
    for iter = 1:num_iters
        [cost, grad] = f(nn_params);
        grad = grad * alpha;
        nn_params = nn_params - grad;
        if mod(iter, 100) == 0
            Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));
            Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));
            pred = predict(Theta1, Theta2, X);
            fprintf('iter: %f accuracy: %f time used:%f\n', iter, mean(double(pred == y)) * 100, toc);
        end
    end
    %time_taken = toc;
    %disp(time_taken);
    matlabpool close;
end
    
