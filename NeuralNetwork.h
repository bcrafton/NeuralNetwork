#include "NNinclude.h"
#include "matrix.h"

void gradient_descent(matrix_t* rolled_theta, unsigned int layer_sizes[], unsigned int num_layers,
		unsigned int num_labels, matrix_t* X, matrix_t* y, double lamda, unsigned int iteration_number);

double NN_cost_function(matrix_t** gradient, matrix_t* rolled_theta, unsigned int layer_sizes[], unsigned int num_layers, unsigned int num_labels, matrix_t* X, matrix_t* y, double lamda);

double accuracy(matrix_list_t* theta, matrix_t* X, matrix_t* y);

matrix_list_t* random_init_weights(unsigned int num_layers, unsigned int layer_sizes[]);


