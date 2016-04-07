#include "neural_network.cuh"

#define ALPHA .5

__device__ void calculate_gradient_at(buffer_t* buffer, int index, matrix_list_t** gradient, matrix_list_t* theta, unsigned int num_layers, unsigned int num_labels,
		matrix_t* X, matrix_t* y, double lamda)
{
	unsigned int i;

	matrix_list_t* local_gradient = device_matrix_list_constructor(buffer, theta->num);
	for(i=0; i<local_gradient->num; i++)
	{
		local_gradient->matrix_list[i] = device_matrix_constructor(buffer, theta->matrix_list[i]->rows, theta->matrix_list[i]->cols);
	}

	matrix_t* temp;
	matrix_t* temp2;
	matrix_t* temp3;

	matrix_list_t* A = device_matrix_list_constructor(buffer, num_layers);
	matrix_list_t* Z = device_matrix_list_constructor(buffer, num_layers-1);
	matrix_list_t* delta = device_matrix_list_constructor(buffer, num_layers-1);

	A->matrix_list[0] = device_row_to_vector(buffer, X, index);
	temp = device_matrix_prepend_col(buffer, A->matrix_list[0], 1.0);

	A->matrix_list[0] = device_matrix_transpose(buffer, temp);

	for(i=0; i<num_layers-1; i++)
	{
		Z->matrix_list[i] = device_matrix_multiply(buffer, theta->matrix_list[i], A->matrix_list[i]);

		temp = device_matrix_sigmoid(buffer, Z->matrix_list[i]);
		A->matrix_list[i+1] = device_matrix_prepend_row(buffer, temp, 1.0);
	}

	temp = device_matrix_remove_row(buffer, A->matrix_list[num_layers-1]);
	A->matrix_list[num_layers-1] = temp;
	
	matrix_t* result_matrix = device_matrix_constructor(buffer, 1, num_labels);
	for(i = 0; i < num_labels; i++)
	{
		if(device_vector_get(y, index) == i)
		{
			device_vector_set(result_matrix, i, 1.0);
		}
	}
	temp = device_matrix_transpose(buffer, result_matrix);
	result_matrix= temp;

	delta->matrix_list[1] = device_matrix_subtract(buffer, A->matrix_list[num_layers-1], result_matrix);
	
	matrix_t* theta_transpose = device_matrix_transpose(buffer, theta->matrix_list[1]);
	temp = device_matrix_multiply(buffer, theta_transpose, delta->matrix_list[1]);

	matrix_t* sig_gradient = device_matrix_sigmoid_gradient(buffer, Z->matrix_list[0]);
	temp2 = device_matrix_prepend_row(buffer, sig_gradient, 1.0);

	temp3 = device_matrix_cell_multiply(buffer, temp, temp2);
	delta->matrix_list[0] = device_matrix_remove_row(buffer, temp3);

	for(i=0; i<num_layers-1; i++)
	{
		matrix_t* A_transpose = device_matrix_transpose(buffer, A->matrix_list[i]);
		temp = device_matrix_multiply(buffer, delta->matrix_list[i], A_transpose);
		temp2 = device_matrix_add(buffer, local_gradient->matrix_list[i], temp);
		local_gradient->matrix_list[i] = temp2;
	}

	*gradient = local_gradient;
}

__global__ void calculate_gradient_kernel(void* gradient, void* memptr, size_t size, matrix_t* rolled_theta, unsigned int num_layers, unsigned int num_labels,
		matrix_t* X, matrix_t* y, double lamda)
{
	unsigned int layer_sizes[][2] = {{25, 401}, {10, 26}};

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < 5000)
	{
		buffer_t* buffer = buffer_constructor(size, memptr+(tid * size));	
		
		matrix_list_t* local_gradient;
		matrix_list_t* theta = device_unroll_matrix_list(buffer, rolled_theta, num_layers-1, layer_sizes);
		
		
		//calculate_gradient_at(buffer, tid, &local_gradient, theta, num_layers, num_labels, X, y, lamda);

		matrix_t* result = (matrix_t*) (gradient + (tid * device_matrix_memory_size(rolled_theta)));

		matrix_t* tmp = device_roll_matrix_list(buffer, theta);
		memcpy(result, tmp, device_matrix_memory_size(rolled_theta));
	}
}

void calculate_gradient(matrix_list_t** gradient, matrix_list_t* theta, unsigned int num_layers, unsigned int num_labels,
		matrix_t* X, matrix_t* y, double lamda)
{
	unsigned int layer_sizes[][2] = {{25, 401}, {10, 26}};

	unsigned int m = X->rows;
	//unsigned int n = X->cols;
	unsigned int i, j;

	void* memptr;
	void* device_gradient;
	matrix_t* device_rolled_theta;
	matrix_t* device_X;
	matrix_t* device_y;
	
	matrix_t* rolled_theta = roll_matrix_list(theta);

	cudaMalloc(&memptr, 262144*5000);
	cudaMalloc(&device_gradient, matrix_memory_size(rolled_theta)*5000);
	cudaMalloc(&device_rolled_theta, matrix_memory_size(rolled_theta));
	cudaMalloc(&device_X, matrix_memory_size(X));
	cudaMalloc(&device_y, matrix_memory_size(y));
	
	cudaMemcpy(device_X, X, matrix_memory_size(X), cudaMemcpyHostToDevice);
	cudaMemcpy(device_y, y, matrix_memory_size(y), cudaMemcpyHostToDevice);
	cudaMemcpy(device_rolled_theta, rolled_theta, matrix_memory_size(rolled_theta), cudaMemcpyHostToDevice);
	
	int block_size = 1024;
	int grid_size = 5000 / block_size;
	if(5000 % block_size)
	{
		grid_size = grid_size + 1;
	}
	
	calculate_gradient_kernel<<<grid_size, block_size>>>(device_gradient, memptr, 262144, device_rolled_theta, num_layers, num_labels, device_X, device_y, lamda);
	
	matrix_t* rolled_gradient = matrix_constructor(rolled_theta->rows, rolled_theta->cols);
	cudaMemcpy(rolled_gradient, device_gradient, matrix_memory_size(rolled_theta), cudaMemcpyDeviceToHost);
	matrix_list_t* gradient_sum = unroll_matrix_list(rolled_gradient, num_layers-1, layer_sizes);	

	matrix_t* temp;
	matrix_t* temp2;
	matrix_t* temp3;
	
	for(i=0; i<num_layers-1; i++)
	{
		temp = matrix_scalar_multiply(gradient_sum->matrix_list[i], 1.0/m);
		temp2 = copy_matrix(theta->matrix_list[i]);
		for(j=0; j<theta->matrix_list[i]->rows; j++)
		{
			matrix_set(temp2, j, 0, 0.0);
		}
		free_matrix(gradient_sum->matrix_list[i]);
		temp3 = matrix_scalar_multiply(temp2, lamda/m);
		gradient_sum->matrix_list[i] = matrix_add(temp, temp3);
		free_matrix(temp);
		free_matrix(temp2);
		free_matrix(temp3);
	}

	*gradient = gradient_sum;

	cudaFree(memptr);
	cudaFree(device_gradient);
	cudaFree(device_rolled_theta);
	cudaFree(device_X);
	cudaFree(device_y);
	
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
		calculate_gradient(&gradient, *theta, num_layers, num_labels, X, y, lamda);
		
		matrix_list_t* tmp;
		tmp = matrix_list_scalar_multiply(gradient, ALPHA);
		free_matrix_list(gradient);
		gradient = tmp;

		tmp = matrix_list_subtract(*theta, gradient);
		free_matrix_list(*theta);
		*theta = tmp;

		free_matrix_list(gradient);

		if((i+1) % 10 == 0)
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
