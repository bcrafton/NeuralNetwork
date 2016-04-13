function accuracy = ex4()
    disp('starting');
    input_layer_size  = 400;  
    hidden_layer_size = 25;   
    num_labels = 10;

    load('ex4data1.mat');
    m = size(X, 1);

    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

    options = optimset('MaxIter', 25);

    lambda = 0.8;

    costFunction = @(p) nnCostFunction(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, X, y, lambda);

    % [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    [nn_params] = gradientDescent(costFunction, initial_nn_params, X, y, input_layer_size, hidden_layer_size, num_labels);

    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));
    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));

    pred = predict(Theta1, Theta2, X);

    %fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
    accuracy = mean(double(pred == y)) * 100;
    