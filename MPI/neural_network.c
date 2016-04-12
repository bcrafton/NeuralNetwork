#include "neural_network.h"
#include "mpi.h"

#define ALPHA .5

static void get_indexes(int problem_size, int num_threads, int tid, int* indexes);

void calculate_gradient_at(int index, matrix_list_t** gradient, matrix_list_t* theta, unsigned int num_layers, unsigned int num_labels,
		matrix_t* X, matrix_t* y, double lamda)
{
	unsigned int i;

	matrix_list_t* local_gradient = matrix_list_constructor(theta->num);
	for(i=0; i<local_gradient->num; i++)
	{
		local_gradient->matrix_list[i] = matrix_constructor(theta->matrix_list[i]->rows, theta->matrix_list[i]->cols);
	}

	matrix_t* temp;
	matrix_t* temp2;
	matrix_t* temp3;

	matrix_list_t* A = matrix_list_constructor(num_layers);
	matrix_list_t* Z = matrix_list_constructor(num_layers-1);
	matrix_list_t* delta = matrix_list_constructor(num_layers-1);

	A->matrix_list[0] = row_to_vector(X, index);
	temp = matrix_prepend_col(A->matrix_list[0], 1.0);
	free_matrix(A->matrix_list[0]);

	A->matrix_list[0] = matrix_transpose(temp);
	free_matrix(temp);

	for(i=0; i<num_layers-1; i++)
	{
		Z->matrix_list[i] = matrix_multiply(theta->matrix_list[i], A->matrix_list[i]);

		temp = matrix_sigmoid(Z->matrix_list[i]);
		A->matrix_list[i+1] = matrix_prepend_row(temp, 1.0);
		free_matrix(temp);
	}

	temp = matrix_remove_row(A->matrix_list[num_layers-1]);
	free_matrix(A->matrix_list[num_layers-1]);
	A->matrix_list[num_layers-1] = temp;

	matrix_t* result = matrix_constructor(1, num_labels);
	for(i = 0; i < num_labels; i++)
	{
		if(vector_get(y, index) == i)
		{
			vector_set(result, i, 1.0);
		}
	}
	temp = matrix_transpose(result);
	free_matrix(result);
	result = temp;

	delta->matrix_list[1] = matrix_subtract(A->matrix_list[num_layers-1], result);
	free_matrix(result);

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

	for(i=0; i<num_layers-1; i++)
	{
		matrix_t* A_transpose = matrix_transpose(A->matrix_list[i]);
		temp = matrix_multiply(delta->matrix_list[i], A_transpose);
		temp2 = matrix_add(local_gradient->matrix_list[i], temp);
		free_matrix(local_gradient->matrix_list[i]);
		local_gradient->matrix_list[i] = temp2;


		free_matrix(A_transpose);
		free_matrix(temp);
	}
	free_matrix_list(A);
	free_matrix_list(Z);
	free_matrix_list(delta);

	*gradient = local_gradient;
}

void calculate_gradient(int rank, int size, matrix_list_t** gradient, matrix_list_t* theta, unsigned int num_layers, unsigned int num_labels,
		matrix_t* X, matrix_t* y, double lamda)
{
	unsigned int m = X->rows;
	//unsigned int n = X->cols;
	
	matrix_list_t* gradient_sum = matrix_list_constructor(theta->num);
	unsigned int i, j;
	for(i=0; i<gradient_sum->num; i++)
	{
		gradient_sum->matrix_list[i] = matrix_constructor(theta->matrix_list[i]->rows, theta->matrix_list[i]->cols);
	}
	
	matrix_list_t* local_gradient;

	int indexes[2];
	get_indexes(m, size, rank, indexes);
	for(i=indexes[0]; i<indexes[1]; i++)
	{
		calculate_gradient_at(i, &local_gradient, theta, num_layers, num_labels, X, y, lamda);
		matrix_list_add2(gradient_sum, local_gradient, gradient_sum);
		free_matrix_list(local_gradient);
	}

	*gradient = gradient_sum;
}


void gradient_descent(matrix_list_t** theta, unsigned int num_layers, unsigned int num_labels,
		double lamda, unsigned int iteration_number)
{
	unsigned int i, j, k;
	unsigned int layer_sizes[][2] = {{25, 401}, {10, 26}};
	double start_time;

	matrix_list_t* gradient = matrix_list_constructor((*theta)->num);
	for(i=0; i<gradient->num; i++)
	{
		gradient->matrix_list[i] = matrix_constructor((*theta)->matrix_list[i]->rows, (*theta)->matrix_list[i]->cols);
	}

	void* buffer = NULL; 

	int rank;
	int size;
	int argc = 0;
	char** argv = NULL;
	
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	start_time = MPI_Wtime();

	if(rank == 0)
	{
		buffer = malloc((sizeof(matrix_t) + sizeof(float) * (layer_sizes[0][0]*layer_sizes[0][1] + layer_sizes[1][0]*layer_sizes[1][1]))*size);
	}

	matrix_t* X = load_from_file("X.csv", 5000, 400);
	matrix_t* tmp = load_from_file("y.csv", 5000, 1);
	matrix_t* y = matrix_transpose(tmp);
	free_matrix(tmp);

	unsigned int m = X->rows;

	for(i=0; i < iteration_number; i++)
	{
		matrix_t* rolled_theta = matrix_constructor(1, layer_sizes[0][0]*layer_sizes[0][1] + layer_sizes[1][0]*layer_sizes[1][1]);
		if(rank == 0)
		{
			rolled_theta = roll_matrix_list(*theta);
		}
		
		MPI_Bcast(rolled_theta, matrix_memory_size(rolled_theta), MPI_CHAR, 0, MPI_COMM_WORLD);
		if(rank != 0)
		{
			*theta = unroll_matrix_list(rolled_theta, num_layers-1, layer_sizes);
		}

		matrix_list_t* local_gradient;
		calculate_gradient(rank, size, &local_gradient, *theta, num_layers, num_labels, X, y, lamda);
		
		matrix_t* rolled_local_gradient = roll_matrix_list(local_gradient);
		
		MPI_Gather(rolled_local_gradient, matrix_memory_size(rolled_local_gradient), MPI_CHAR, buffer, matrix_memory_size(rolled_local_gradient), MPI_CHAR, 0, MPI_COMM_WORLD);
		
		if(rank == 0)
		{
			int memory_size = matrix_memory_size(rolled_local_gradient);
			for(j=0; j<size; j++)
			{
				rolled_local_gradient = (matrix_t*) (buffer + j*memory_size);
				local_gradient = unroll_matrix_list(rolled_local_gradient, num_layers-1, layer_sizes);
				matrix_list_add2(gradient, local_gradient, gradient);
			}

			matrix_t* temp;
			matrix_t* temp2;
			matrix_t* temp3;
			
			for(j=0; j<num_layers-1; j++)
			{
				
				temp = matrix_scalar_multiply(gradient->matrix_list[j], 1.0/m);
				temp2 = copy_matrix((*theta)->matrix_list[j]);
				for(k=0; k<(*theta)->matrix_list[j]->rows; k++)
				{
					matrix_set(temp2, k, 0, 0.0);
				}
				
				free_matrix(gradient->matrix_list[j]);
				temp3 = matrix_scalar_multiply(temp2, lamda/m);
				gradient->matrix_list[j] = matrix_add(temp, temp3);

				free_matrix(temp);
				free_matrix(temp2);
				free_matrix(temp3);
			}
			

			matrix_list_t* tmp;
			tmp = matrix_list_scalar_multiply(gradient, ALPHA);
			free_matrix_list(gradient);
			gradient = tmp;

			tmp = matrix_list_subtract(*theta, gradient);
			free_matrix_list(*theta);
			*theta = tmp;

			for(j=0; j<gradient->num; j++)
			{
				set_matrix(gradient->matrix_list[j], 0.0);
			}

			if((i+1) % 100 == 0)
			{
				double cpu_time_used = MPI_Wtime() - start_time;
				printf("iteration #%d, accuracy: %f, time used: %f\n", i+1, accuracy(*theta, X, y), cpu_time_used);
			}
		}
	}
	MPI_Finalize();
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

void get_indexes(int problem_size, int num_threads, int tid, int* indexes)
{
	int block_size = problem_size / num_threads;
	int left_over = problem_size % num_threads;
	int start_index;
	int end_index;
	if(tid >= left_over)
	{
		start_index = block_size * tid + left_over;
	}
	else
	{
		start_index = block_size * tid + tid;
	}

	if(tid >= left_over)
	{
		end_index = start_index + block_size;
	}
	else
	{
		end_index = start_index + block_size + 1;
	}
	indexes[0] = start_index;
	indexes[1] = end_index;
}
