#include "NeuralNetwork.h"

#define ALPHA .5

__global__ void calculate_gradient(matrix_list_t* gradient, matrix_list_t* theta, unsigned int num_layers, unsigned int num_labels,
		matrix_t* X, matrix_t* y, double lamda, buffer_t* buffer)
{
	set_buffer(buffer);

	unsigned int m = X->rows;
	//unsigned int n = X->cols;

	matrix_list_t* theta_gradient = device_matrix_list_constructor(theta->num);
	unsigned int i, j;
	for(i=0; i<theta_gradient->num; i++)
	{
		theta_gradient->matrix_list[i] = device_matrix_constructor(theta->matrix_list[i]->rows, theta->matrix_list[i]->cols);
	}
	
	int tid = ;
	
	matrix_t* temp;
	matrix_t* temp2;
	matrix_t* temp3;
	
	matrix_list_t* A = device_matrix_list_constructor(num_layers);
	matrix_list_t* Z = device_matrix_list_constructor(num_layers-1);
	matrix_list_t* delta = device_matrix_list_constructor(num_layers-1);

	A->matrix_list[0] = device_row_to_vector(X, tid);
	temp = device_matrix_prepend_col(A->matrix_list[0], 1.0);
	device_free_matrix(A->matrix_list[0]);

	A->matrix_list[0] = device_matrix_transpose(temp);
	device_free_matrix(temp);

	for(j=0; j<num_layers-1; j++)
	{
		Z->matrix_list[j] = device_matrix_multiply(theta->matrix_list[j], A->matrix_list[j]);

		temp = device_matrix_sigmoid(Z->matrix_list[j]);
		A->matrix_list[j+1] = device_matrix_prepend_row(temp, 1.0);
		device_free_matrix(temp);
	}

	temp = device_matrix_remove_row(A->matrix_list[num_layers-1]);
	device_free_matrix(A->matrix_list[num_layers-1]);
	A->matrix_list[num_layers-1] = temp;

	matrix_t* class = device_matrix_constructor(1, num_labels);
	for(j = 0; j < num_labels; j++)
	{
		if(device_vector_get(y, tid) == j)
		{
			device_vector_set(class, j, 1.0);
		}
	}
	temp = device_matrix_transpose(class);
	device_free_matrix(class);
	class = temp;

	delta->matrix_list[1] = device_matrix_subtract(A->matrix_list[num_layers-1], class);
	device_free_matrix(class);

	matrix_t* theta_transpose = device_matrix_transpose(theta->matrix_list[1]);
	temp = device_matrix_multiply(theta_transpose, delta->matrix_list[1]);

	matrix_t* sig_gradient = device_matrix_sigmoid_gradient(Z->matrix_list[0]);
	temp2 = device_matrix_prepend_row(sig_gradient, 1.0);

	temp3 = device_matrix_cell_multiply(temp, temp2);
	delta->matrix_list[0] = device_matrix_remove_row(temp3);

	device_free_matrix(temp);
	device_free_matrix(temp2);
	device_free_matrix(temp3);
	device_free_matrix(sig_gradient);
	device_free_matrix(theta_transpose);

	for(j=0; j<num_layers-1; j++)
	{
		matrix_t* A_transpose = device_matrix_transpose(A->matrix_list[j]);
		temp = device_matrix_multiply(delta->matrix_list[j], A_transpose);
		temp2 = device_matrix_add(theta_gradient->matrix_list[j], temp);
		device_free_matrix(theta_gradient->matrix_list[j]);
		theta_gradient->matrix_list[j] = temp2;

		device_free_matrix(A_transpose);
		device_free_matrix(temp);
	}
	device_free_matrix_list(A);
	device_free_matrix_list(Z);
	device_free_matrix_list(delta);
}

void NN_cost_function(matrix_list_t** gradient, matrix_list_t* theta, unsigned int num_layers, unsigned int num_labels,
		matrix_t* X, matrix_t* y, double lamda)
{
	unsigned int m = X->rows;
	//unsigned int n = X->cols;

	matrix_list_t* theta_gradient = matrix_list_constructor(theta->num);
	unsigned int i, j;
	for(i=0; i<theta_gradient->num; i++)
	{
		theta_gradient->matrix_list[i] = matrix_constructor(theta->matrix_list[i]->rows, theta->matrix_list[i]->cols);
	}
	
	buffer_t* device_buffer;
	matrix_t* device_theta;
	matrix_t* device_X;
	matrix_t* device_y;
	matrix_t* device_theta_gradient;
	
	cudaMalloc(&device_buffer, 200000 * 5000);
	cudaMalloc(&device_theta_gradient, matrix_list_memory_size(theta));
	cudaMalloc(&device_theta, matrix_list_memory_size(theta));
	cudaMalloc(&device_X, matrix_list_memory_size(X));
	cudaMalloc(&device_y, matrix_list_memory_size(y));
	
	cudaMemcpy(device_y, y, cudaMemcpyHostToDevice);
	cudaMemcpy(device_X, x, cudaMemcpyHostToDevice);
	cudaMemcpy(device_theta, theta, cudaMemcpyHostToDevice);
	
	int block_size = 1024;
	int grid_size = 5000 / block_size;
	if(5000 % block_size)
	{
		grid_size = grid_size + 1;
	}
	
	calculate_gradient<<<grid_size, block_size>>>(device_theta_gradient, device_theta, num_layers, num_labels, device_X, device_y, lamda, device_buffer);
	
	// perform the reduction here.
	
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

	*gradient = theta_gradient;
}


void gradient_descent(matrix_list_t** theta, unsigned int num_layers, unsigned int num_labels, matrix_t* X, matrix_t* y,
		double lamda, unsigned int iteration_number)
{
	clock_t start, end;
	double cpu_time_used;
	start = clock();

	matrix_list_t* gradient;

	unsigned int i;
	for(i=0; i < iteration_number; i++)
	{
		NN_cost_function(&gradient, *theta, num_layers, num_labels, X, y, lamda);

		matrix_list_t* tmp;
		tmp = matrix_list_scalar_multiply(gradient, ALPHA);
		free_matrix_list(gradient);
		gradient = tmp;

		tmp = matrix_list_subtract(*theta, gradient);
		free_matrix_list(*theta);
		*theta = tmp;

		free_matrix_list(gradient);

		if((i+1) % 1 == 0)
		{
			end = clock();
			cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
			printf("iteration #%d, accuracy: %f, time used: %f\n", i+1, accuracy(*theta, X, y), cpu_time_used);
		}
	}
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
