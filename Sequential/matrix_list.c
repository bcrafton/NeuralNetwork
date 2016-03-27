#include "matrix.h"

matrix_list_t* matrix_list_constructor(unsigned int num)
{
	buffer_t* buffer = get_buffer();

	matrix_list_t* list = (matrix_list_t*)buffer_malloc(buffer, sizeof(matrix_list_t));
	list->num = num;
	list->matrix_list = (matrix_t**)buffer_malloc(buffer, sizeof(matrix_t*) * num);
	return list;
}

void free_matrix_list(matrix_list_t* m)
{
	/*
	assert(m != NULL);
	int i;
	for(i=0; i<m->num; i++)
	{
		free_matrix(m->matrix_list[i]);
	}
	free(m);
	*/
}

matrix_list_t* matrix_list_add(matrix_list_t* m1, matrix_list_t* m2)
{
	assert(m1->num == m2->num);
	matrix_list_t* m = matrix_list_constructor(m1->num);

	int i;
	for(i=0; i<m1->num; i++)
	{
		m->matrix_list[i] = matrix_add(m1->matrix_list[i], m2->matrix_list[i]);
	}
	return m;
}

matrix_list_t* matrix_list_subtract(matrix_list_t* m1, matrix_list_t* m2)
{
	assert(m1->num == m2->num);
	matrix_list_t* m = matrix_list_constructor(m1->num);

	int i;
	for(i=0; i<m1->num; i++)
	{
		m->matrix_list[i] = matrix_subtract(m1->matrix_list[i], m2->matrix_list[i]);
	}
	return m;
}

matrix_list_t* matrix_list_scalar_multiply(matrix_list_t* m1, float scalar)
{
	matrix_list_t* m = matrix_list_constructor(m1->num);

	int i;
	for(i=0; i<m1->num; i++)
	{
		m->matrix_list[i] = matrix_scalar_multiply(m1->matrix_list[i], scalar);
	}
	return m;
}

unsigned int matrix_list_memory_size(matrix_list_t* m)
{
	unsigned int memory_size = sizeof(matrix_list_t);
	unsigned int i;
	for(i=0; i<m->num; i++)
	{
		memory_size += matrix_memory_size(m->matrix_list[i]);
	}
	return memory_size;
}
