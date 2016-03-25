#include "NeuralNetwork.h"

#define ALPHA .5

__global__ void NN_cost_function(matrix_list_t* theta_gradient, matrix_t* rolled_theta, unsigned int* layer_sizes, unsigned int num_layers,
		unsigned int num_labels, matrix_t* X, matrix_t* y, float lamda)
{
	unsigned int theta_sizes[][2] = {{25, 401}, {10, 26}};

	matrix_list_t* theta = unroll_matrix_list(rolled_theta, num_layers-1, theta_sizes);

	unsigned int m = X->rows;
	//unsigned int n = X->cols;

	matrix_list_t* theta_gradient = matrix_list_constructor(theta->num);
	unsigned int i, j;
	for(i=0; i<theta_gradient->num; i++)
	{
		theta_gradient->matrix_list[i] = matrix_constructor(theta->matrix_list[i]->rows, theta->matrix_list[i]->cols);
	}

	matrix_t* temp;
	matrix_t* temp2;
	matrix_t* temp3;
	for(i=0; i<m; i++)
	{
		matrix_list_t* A = matrix_list_constructor(num_layers);
		matrix_list_t* Z = matrix_list_constructor(num_layers-1);
		matrix_list_t* delta = matrix_list_constructor(num_layers-1);

		A->matrix_list[0] = row_to_vector(X, i);
		temp = matrix_prepend_col(A->matrix_list[0], 1.0);
		free_matrix(A->matrix_list[0]);

		A->matrix_list[0] = matrix_transpose(temp);
		free_matrix(temp);

		for(j=0; j<num_layers-1; j++)
		{
			Z->matrix_list[j] = matrix_multiply(theta->matrix_list[j], A->matrix_list[j]);

			temp = matrix_sigmoid(Z->matrix_list[j]);
			A->matrix_list[j+1] = matrix_prepend_row(temp, 1.0);
			free_matrix(temp);
		}

		temp = matrix_remove_row(A->matrix_list[num_layers-1]);
		free_matrix(A->matrix_list[num_layers-1]);
		A->matrix_list[num_layers-1] = temp;

		matrix_t* class = matrix_constructor(1, num_labels);
		for(j = 0; j < num_labels; j++)
		{
			if(vector_get(y, i) == j)
			{
				vector_set(class, j, 1.0);
			}
		}
		temp = matrix_transpose(class);
		free_matrix(class);
		class = temp;

		delta->matrix_list[1] = matrix_subtract(A->matrix_list[num_layers-1], class);
		free_matrix(class);

		matrix_t* theta_transpose = matrix_transpose(theta->matrix_list[1]);
		temp = matrix_multiply(theta_transpose, delta->matrix_list[1]);

		matrix_t* sig_gradient = matrix_sigmoid_gradient(Z->matrix_list[0]);
		temp2 = matrix_prepend_row(sig_gradient, 1.0);

		temp3 = matrix_cell_multiply(temp, temp2);
		delta->matrix_list[0] = matrix_remove_row(temp3);

		free_matrix(temp);
		free_matrix(temp2);
		free_matrix(temp3);
		free_matrix(sig_gradient);
		free_matrix(theta_transpose);

		for(j=0; j<num_layers-1; j++)
		{
			matrix_t* A_transpose = matrix_transpose(A->matrix_list[j]);
			temp = matrix_multiply(delta->matrix_list[j], A_transpose);
			temp2 = matrix_add(theta_gradient->matrix_list[j], temp);
			free_matrix(theta_gradient->matrix_list[j]);
			theta_gradient->matrix_list[j] = temp2;


			free_matrix(A_transpose);
			free_matrix(temp);
		}
		free_matrix_list(A);
		free_matrix_list(Z);
		free_matrix_list(delta);
	}
	
	for(i=0; i<num_layers-1; i++)
	{
		temp = matrix_scalar_multiply(theta_gradient->matrix_list[i], 1.0/m);
		temp2 = copy_matrix(theta->matrix_list[i]);
		for(j=0; j<theta->matrix_list[i]->rows; j++)
		{
			matrix_set(temp2, j, 0, 0.0);
		}
		free_matrix(theta_gradient->matrix_list[i]);
		temp3 = matrix_scalar_multiply(temp2, lamda/m);
		theta_gradient->matrix_list[i] = matrix_add(temp, temp3);
		free_matrix(temp);
		free_matrix(temp2);
		free_matrix(temp3);
	}

	*gradient = roll_matrix_list(theta_gradient);

	free_matrix_list(theta);
	free_matrix_list(theta_gradient);

	return 0.0;
}

//todo: replace loop rolling and unrolling with matrix list arithmetic, WAY easier and simpler that way.
void gradient_descent(matrix_t* rolled_theta, unsigned int layer_sizes[], unsigned int num_layers,
		unsigned int num_labels, matrix_t* X, matrix_t* y, float lamda, unsigned int iteration_number)
{
	int block_size = 1024;
	int grid_size = 5000 / block_size;
	if(vector_size % block_size)
	{
		grid_size = grid_size + 1;
	}
	
	unsigned int theta_sizes[][2] = {{25, 401}, {10, 26}};
	
	// the 5 things that need to be passed to device with cudaMalloc, rest by value
	matrix_t* device_X;
	matrix_t* device_y;
	matrix_list_t* device_theta;
	unsigned int* device_theta_sizes;
	matrix_list_t* device_theta_gradient;
	
	cudaMalloc(&device_X, matrix_memory_size(X));
	cudaMalloc(&device_y, matrix_memory_size(y));
	cudaMalloc(&device_theta, matrix_list_memory_size(theta));
	cudaMalloc(&device_theta_sizes, sizeof(theta_sizes)); // make sure this is the actual size
	cudaMalloc(&device_theta_gradient, matrix_list_memory_size(theta)*5000);
	
	cudaMemcpy( device_X, X, matrix_memory_size(X), cudaMemcpyHostToDevice);
	cudaMemcpy( device_y, y, matrix_memory_size(y), cudaMemcpyHostToDevice);
	cudaMemcpy( device_theta, theta, matrix_list_memory_size(theta), cudaMemcpyHostToDevice);
	cudaMemcpy( device_theta_sizes, theta_sizes, sizeof(theta_sizes), cudaMemcpyHostToDevice);
	
	unsigned int i;
	for(i=0; i < iteration_number; i++)
	{
		// this shud not just be a kernel.
		// the cost function will wrap the kernel and then grab the resulting gradient and give it to gradient descent.
		NN_cost_function<<<grid_size, block_size>>>(device_theta_gradient, device_rolled_theta, 
			device_layer_sizes, num_layers, num_labels, device_X, device_y, lamda);

		// NOT ACTUALLY USING THE gradient
		matrix_t* tmp;
		tmp = matrix_scalar_multiply(gradient, ALPHA);
		free_matrix(gradient);
		gradient = tmp;

		tmp = matrix_subtract(rolled_theta, gradient);
		free_matrix(rolled_theta);
		rolled_theta = tmp;

		free_matrix(gradient);

		if((i+1) % 100 == 0)
		{
			matrix_list_t* theta = unroll_matrix_list(rolled_theta, num_layers-1, theta_sizes);
			printf("iteration #%d, accuracy: %f, time used: %f\n", i+1, accuracy(theta, X, y), cpu_time_used);
			free_matrix_list(theta);
		}
	}
	free_matrix(rolled_theta);
}


matrix_list_t* random_init_weights(unsigned int num_layers, unsigned int layer_sizes[])
{
	srand(time(NULL));

	matrix_list_t* theta = matrix_list_constructor(num_layers-1);
	unsigned int i, j, k;
	for(i = 0; i<num_layers-1; i++)
	{
		theta->matrix_list[i] = matrix_constructor(layer_sizes[i+1], layer_sizes[i]+1);
		for(j=0; j<theta->matrix_list[i]->rows; j++)
		{
			for(k = 0; k<theta->matrix_list[i]->cols; k++)
			{
				double random_double = ((double)(rand() % 1000)) / (double)1000;
				matrix_set(theta->matrix_list[i], j, k, random_double * 2 * .12 - .12);
			}
		}
	}
	return theta;
}

double accuracy(matrix_list_t* theta, matrix_t* X, matrix_t* y)
{
	assert(theta->num == 2);
	matrix_t* theta_transpose, *temp, *temp2;

	theta_transpose = matrix_transpose(theta->matrix_list[0]);
	temp = matrix_prepend_col(X, 1.0);
	temp2 = matrix_multiply(temp, theta_transpose);
	matrix_t* h1 = matrix_sigmoid(temp2);

	free_matrix(theta_transpose);
	free_matrix(temp);
	free_matrix(temp2);

	theta_transpose = matrix_transpose(theta->matrix_list[1]);
	temp = matrix_prepend_col(h1, 1.0);
	temp2 = matrix_multiply(temp, theta_transpose);
	matrix_t* h2 = matrix_sigmoid(temp2);

	free_matrix(theta_transpose);
	free_matrix(temp);
	free_matrix(temp2);

	assert(h2->rows == 5000 && h2->cols == 10);
	matrix_t* p = matrix_constructor(1, 5000);
	int i, j;

	for(i = 0; i<h2->rows; i++)
	{
		double max = 0.0;
		unsigned char first = 1;
		for(j=0; j<h2->cols; j++)
		{
			if(matrix_get(h2, i, j) > max || first == 1)
			{
				vector_set(p, i, j);
				max = matrix_get(h2, i, j);
				first = 0;
			}
		}
	}
	double count = 0;
	for(i=0; i<5000; i++)
	{
		if(vector_get(y, i) == vector_get(p, i))
			count = count + 1;
	}

	free_matrix(p);
	free_matrix(h1);
	free_matrix(h2);
	
	return count/5000;
}