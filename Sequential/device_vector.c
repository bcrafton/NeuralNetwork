#include "matrix.h"

__device__ float device_vector_get(matrix_t* v, unsigned int x)
{
	assert(v->rows == 1);
	assert(v->cols >= 0);
	return v->matrix[x];
}

__device__ void device_vector_set(matrix_t* v, unsigned int x, float value)
{
	assert(v->rows == 1);
	assert(v->cols >= 0);
	v->matrix[x] = value;
}
