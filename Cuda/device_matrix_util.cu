#include "matrix.cuh"

#define ROW_INDEX 0
#define COL_INDEX 1
#define NUM_INDEXES 2

__device__ matrix_t* device_roll_matrix_list(buffer_t* buffer, matrix_list_t* list)
{
	unsigned int i;
	//assert(list != NULL);
	//for(i=0; i<list->num; i++)
	//{
	//	assert(list->matrix_list[i] != NULL);
	//}

	unsigned int vector_size=0;
	for(i=0; i<list->num; i++)
	{
		vector_size += list->matrix_list[i]->rows * list->matrix_list[i]->cols;
	}
	matrix_t* vector = device_matrix_constructor(buffer, 1, vector_size);
	float* current_index = vector->matrix;

	for(i=0; i<list->num; i++)
	{
		unsigned int matrix_size = list->matrix_list[i]->rows * list->matrix_list[i]->cols;
		memcpy(current_index, list->matrix_list[i]->matrix, matrix_size * sizeof(float));
		current_index = current_index + matrix_size;
	}
	return vector;
}

__device__ matrix_list_t* device_unroll_matrix_list(buffer_t* buffer, matrix_t* vector, int num, unsigned int sizes[][NUM_INDEXES])
{
	//assert(vector != NULL);

	matrix_list_t* list = device_matrix_list_constructor(buffer, num);

	float* current_index = vector->matrix;
	unsigned int i;

	for(i=0; i<num; i++)
	{
		list->matrix_list[i] = device_matrix_constructor(buffer, sizes[i][ROW_INDEX], sizes[i][COL_INDEX]);

		unsigned int matrix_size = sizes[i][ROW_INDEX] * sizes[i][COL_INDEX];
		memcpy(list->matrix_list[i]->matrix, current_index, matrix_size * sizeof(float));
		current_index = current_index + matrix_size;
	}

	return list;
}


