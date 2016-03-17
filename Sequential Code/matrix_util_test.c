#include "matrix.h"

#define NUM_MATRICES 3
#define MATRIX_SIZE 5
#define ROW_INDEX 0
#define COL_INDEX 1

const int matrix_sizes[][2] = {{1,5},{1,5},{5,1}};
const int roll_answer[3][5] = {
		{0.0, 0.0, 0.0, 0.0, 0.0},
		{1.0, 1.0, 1.0, 1.0, 1.0},
		{2.0, 2.0, 2.0, 2.0, 2.0}};

unsigned int matrix_util_test()
{
	unsigned int success;

	success = test_roll_unroll_matrices();
	assert(success);
	printf("matrix roll-unroll pass\n");

	return 1;
}

unsigned int test_roll_unroll_matrices()
{
	matrix_list_t* list = matrix_list_constructor(NUM_MATRICES);
	int i, j;
	for(i = 0; i < NUM_MATRICES; i++)
	{
		list->matrix_list[i] = matrix_constructor(matrix_sizes[i][ROW_INDEX], matrix_sizes[i][COL_INDEX]);
		set_matrix(list->matrix_list[i], (double)i);
	}

	matrix_t* roll = roll_matrix_list(list);
	print_matrix(roll);

	for(i = 0; i < NUM_MATRICES; i++)
	{
		for(j = 0; j < MATRIX_SIZE; j++)
		{
			if(roll->matrix[i*MATRIX_SIZE + j] != roll_answer[i][j])
			{  return 0;  }
		}
	}

	free(list);
	list = unroll_matrix_list(roll, NUM_MATRICES, matrix_sizes);

	for(i = 0; i < NUM_MATRICES; i++)
	{
		print_matrix(list->matrix_list[i]);
		for(j = 0; j < MATRIX_SIZE; j++)
		{
			if(list->matrix_list[i]->matrix[j] != roll_answer[i][j])
			{  return 0;  }
		}
	}
	return 1;
}
