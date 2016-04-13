function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
Theta = cell(2, 1);
Theta_grad = cell(2,1);

num_layers = 3;
                               
Theta{1} = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta{2} = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
             
for i = 1:num_layers-1
    Theta_grad{i} = zeros(size(Theta{i}));
end

m = size(X, 1);
         
J = 0;

spmd
    start = (labindex-1)*(m/numlabs) + 1;
    last = labindex * (m/numlabs);
    
    Theta_grad_local = cell(2,1);

    for i = 1:num_layers-1
        Theta_grad_local{i} = zeros(size(Theta{i}));
    end
    
    for t = start:last
        A = cell(3,1);
        Z = cell(3,1);
        delta = cell(3,1);

        A{1} = [1; X(t,:)'];
        for j = 2:num_layers
            Z{j} = Theta{j-1} * A{j-1};
            A{j} = [1; sigmoid(Z{j})];
        end

        A{num_layers} = A{num_layers}(2:end);

        yy = ([1:num_labels]==y(t))';


        delta{num_layers} = A{num_layers} - yy;


        for i = num_layers-1 : -1 : 2
            delta{i} = (Theta{i}' * delta{i+1}) .* [1; sigmoidGradient(Z{i})];
            delta{i} = delta{i}(i:end);
        end

        for i = 1:num_layers-1
            Theta_grad_local{i} = Theta_grad_local{i} + delta{i+1} * A{i}';
        end
    end
    Theta_grad{1} = gplus(Theta_grad_local{1});
    Theta_grad{2} = gplus(Theta_grad_local{2});
end

Theta_grad = Theta_grad{1};

for i = 1:num_layers-1
    Theta_grad{i} = (1/m) * Theta_grad{i} + (lambda/m) * [zeros(size(Theta{i}, 1), 1) Theta{i}(:,2:end)];
end

%disp((lambda/m) * [zeros(size(Theta{2}, 1), 1) Theta{2}(:,2:end)]);

grad = [Theta_grad{1}(:); Theta_grad{2}(:)];

end