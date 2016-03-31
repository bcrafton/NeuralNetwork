#include "matrix.cuh"

__device__ matrix_list_t* device_matrix_list_constructor(buffer_t* buffer, unsigned int num)
{
	matrix_list_t* list = (matrix_list_t*)buffer_malloc(buffer, sizeof(matrix_list_t));
	list->num = num;
	list->matrix_list = (matrix_t**)buffer_malloc(buffer, sizeof(matrix_t*) * num);
	return list;
}

__device__ matrix_list_t* device_matrix_list_add(buffer_t* buffer, matrix_list_t* m1, matrix_list_t* m2)
{
	//assert(m1->num == m2->num);
	matrix_list_t* m = device_matrix_list_constructor(buffer, m1->num);

	int i;
	for(i=0; i<m1->num; i++)
	{
		m->matrix_list[i] = device_matrix_add(buffer, m1->matrix_list[i], m2->matrix_list[i]);
	}
	return m;
}

__device__ matrix_list_t* device_matrix_list_subtract(buffer_t* buffer, matrix_list_t* m1, matrix_list_t* m2)
{
	//assert(m1->num == m2->num);
	matrix_list_t* m = device_matrix_list_constructor(buffer, m1->num);

	int i;
	for(i=0; i<m1->num; i++)
	{
		m->matrix_list[i] = device_matrix_subtract(buffer, m1->matrix_list[i], m2->matrix_list[i]);
	}
	return m;
}

__device__ matrix_list_t* device_matrix_list_scalar_multiply(buffer_t* buffer, matrix_list_t* m1, float scalar)
{
	matrix_list_t* m = device_matrix_list_constructor(buffer, m1->num);

	int i;
	for(i=0; i<m1->num; i++)
	{
		m->matrix_list[i] = device_matrix_scalar_multiply(buffer, m1->matrix_list[i], scalar);
	}
	return m;
}

__device__ void device_free_matrix_list(matrix_list_t* m)
{
}