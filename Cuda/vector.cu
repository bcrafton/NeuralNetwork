#include "matrix.cuh"

float vector_get(matrix_t* v, unsigned int x)
{
	assert(v->rows == 1);
	//assert(v->cols >= 0);
	return v->matrix[x];
}

void vector_set(matrix_t* v, unsigned int x, float value)
{
	assert(v->rows == 1);
	//assert(v->cols >= 0);
	v->matrix[x] = value;
}
