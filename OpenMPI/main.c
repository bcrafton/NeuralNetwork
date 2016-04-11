#include "nn_include.h"
#include "neural_network.h"

#define MATRIX_TEST 0
#define MATRIX_UTIL_TEST 0

int main(int argc,char **argv) {
#if (MATRIX_TEST)
	matrix_test();
#elif (MATRIX_UTIL_TEST)
	matrix_util_test();
#else
	//unsigned int layer_sizes[] = {400, 25, 10};
	unsigned int num_layers = 3;
	unsigned int num_labels = 10;
	double lambda = 0.8;
	
	/*
	if(argc < 2)
	{
		printf("need exactly 1 argument for vector length\n");
		return 0;
	}
    unsigned int iteration_number = atoi(argv[1]);
   	*/

	unsigned int iteration_number = 100;

	matrix_list_t* theta = matrix_list_constructor(2);
	//theta->matrix_list[0] = matrix_random(25, 401, .12);
	//theta->matrix_list[1] = matrix_random(10, 26, .12);
	theta->matrix_list[0] = load_from_file("theta1.csv", 25, 401);
	theta->matrix_list[1] = load_from_file("theta2.csv", 10, 26);
	
	assert(theta->num == 2);
	assert(theta->matrix_list[0]->rows == 25 && theta->matrix_list[0]->cols == 401);
	assert(theta->matrix_list[1]->rows == 10 && theta->matrix_list[1]->cols == 26);
	/*
	matrix_t* X = load_from_file("X.csv", 5000, 400);
	matrix_t* tmp = load_from_file("y.csv", 5000, 1);
	matrix_t* y = matrix_transpose(tmp);
	free_matrix(tmp);
	*/
	gradient_descent(&theta, num_layers, num_labels, lambda, iteration_number);

	//free_matrix(X);
	//free_matrix(y);
	free_matrix_list(theta);
#endif
	return 1;
}
