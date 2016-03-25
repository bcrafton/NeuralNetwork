#include "matrix.h"

__device__ matrix_t* device_matrix_constructor(unsigned int rows, unsigned int cols)
{
	assert(rows > 0 && cols > 0);

	matrix_t* m = (matrix_t*)malloc(sizeof(matrix_t) + sizeof(float) * rows * cols);
	
	assert(m != NULL);

	m->rows = rows;
	m->cols = cols;
	set_matrix(m, 0.0);

	return m;
}

__device__ float device_matrix_get(matrix_t* m, unsigned int x, unsigned int y)
{
	assert(m != NULL);
	assert(x >= 0 && x < m->rows && y >= 0 && y < m->cols);
	return (m->matrix[x * m->cols + y]);
}

__device__ void device_matrix_set(matrix_t* m, unsigned int x, unsigned int y, float value)
{
	assert(m != NULL);
	assert(x >= 0 && x < m->rows && y >= 0 && y < m->cols);
	m->matrix[x * m->cols + y] = value;
}

__device__ matrix_t* device_matrix_add(matrix_t* m1, matrix_t* m2)
{
	assert(m1 != NULL && m2 != NULL);
	assert(m1->rows > 0 && m2->rows > 0 && m1->cols > 0 && m2->cols > 0);
	assert(m1->rows == m2->rows && m1->cols == m2->cols);

	matrix_t* sum = matrix_constructor(m1->rows, m1->cols);

	int i, j;
	for(i=0; i<m1->rows; i++)
	{
		for(j=0; j<m1->cols; j++)
		{
			matrix_set(sum, i, j, matrix_get(m1, i, j) + matrix_get(m2, i, j));
		}
	}
	return sum;
}

__device__ matrix_t* device_matrix_subtract(matrix_t* m1, matrix_t* m2)
{
	assert(m1 != NULL && m2 != NULL);
	assert(m1->rows > 0 && m2->rows > 0 && m1->cols > 0 && m2->cols > 0);
	assert(m1->rows == m2->rows && m1->cols == m2->cols);

	matrix_t* difference = matrix_constructor(m1->rows, m1->cols);

	int i, j;
	for(i=0; i<m1->rows; i++)
	{
		for(j=0; j<m1->cols; j++)
		{
			matrix_set(difference, i, j, matrix_get(m1, i, j) - matrix_get(m2, i , j));
		}
	}
	return difference;
}

__device__ matrix_t* device_matrix_multiply(matrix_t* m1, matrix_t* m2)
{
	if(!(m1->rows > 0 && m2->rows > 0 && m1->cols > 0 && m2->cols > 0))
	{
		printf("%d %d %d %d", m1->rows, m2->rows, m1->cols, m2->cols);
	}
	assert(m1 != NULL && m2 != NULL);
	assert(m1->rows > 0 && m2->rows > 0 && m1->cols > 0 && m2->cols > 0);
	assert(m1->cols == m2->rows);

	matrix_t* product = matrix_constructor(m1->rows, m2->cols);

	int i, j, k;
	for(i=0; i<product->rows; i++)
	{
		for(j=0; j<product->cols; j++)
		{
			for(k=0; k<m1->cols; k++)
			{
				matrix_set(product, i, j, matrix_get(product, i, j) + matrix_get(m1, i, k) * matrix_get(m2, k, j));
			}
		}
	}
	return product;
}

__device__ matrix_t* device_matrix_scalar_multiply(matrix_t* m, float scalar)
{
	assert(m!= NULL);
	assert(m->rows > 0 && m->cols > 0);

	matrix_t* product = matrix_constructor(m->rows, m->cols);

	int i, j;
	for(i=0; i<m->rows; i++)
	{
		for(j=0; j<m->cols; j++)
		{
			matrix_set(product, i, j, matrix_get(m, i, j) * scalar);
		}
	}
	return product;
}

__device__ matrix_t* device_matrix_transpose(matrix_t* m)
{
	assert(m!= NULL);
	assert(m->rows > 0 && m->cols > 0);

	matrix_t* transpose = copy_matrix(m);
	transpose->rows = m->cols;
	transpose->cols = m->rows;
	int i, j;
	for(i=0; i<m->rows; i++)
	{
		for(j=0; j<m->cols; j++)
		{
			matrix_set(transpose, j, i, matrix_get(m, i, j));
		}
	}
	return transpose;
}


__device__ void device_set_matrix(matrix_t* m, float val)
{
	assert(m != NULL);
	assert(m->rows > 0 && m->cols > 0);

	int i, j;
	for(i=0; i<m->rows; i++)
	{
		for(j=0; j<m->cols; j++)
		{
			matrix_set(m, i, j, val);
		}
	}
}

__device__ void device_set_matrix_index(matrix_t* m)
{
	assert(m != NULL);
	assert(m->rows > 0 && m->cols > 0);

	int i, j;
	for(i=0; i<m->rows; i++)
	{
		for(j=0; j<m->cols; j++)
		{
			matrix_set(m, i, j, i * m->cols + j);
		}
	}
}

__device__ matrix_t* device_copy_matrix(matrix_t* m)
{
	matrix_t* copy = matrix_constructor(m->rows, m->cols);
	memcpy(copy->matrix, m->matrix, sizeof(float)*m->rows*m->cols);
	return copy;
}

__device__ matrix_list_t* device_matrix_list_constructor(unsigned int num)
{
	matrix_list_t* list = (matrix_list_t*)malloc(sizeof(matrix_list_t));
	list->num = num;
	list->matrix_list = (matrix_t**)malloc(sizeof(matrix_t*) * num);
	return list;
}

__device__ void device_free_matrix(matrix_t* m)
{
	assert(m != NULL);
	assert(m->matrix != NULL);
	free(m->matrix);
	free(m);
}

__device__ void device_free_matrix_list(matrix_list_t* m)
{
	assert(m != NULL);
	int i;
	for(i=0; i<m->num; i++)
	{
		free_matrix(m->matrix_list[i]);
	}
	free(m);
}

__device__ matrix_t* device_matrix_sigmoid(matrix_t* m)
{
	matrix_t* copy = copy_matrix(m);
	int i, j;
	for(i=0; i<m->rows; i++)
	{
		for(j=0; j<m->cols; j++)
		{
			matrix_set(copy, i, j, 1.0 / (1.0 + exp(-1.0 * matrix_get(copy, i, j))));
		}
	}
	return copy;
}

__device__ matrix_t* device_matrix_sigmoid_gradient(matrix_t* m)
{
	float sig;
	matrix_t* copy = copy_matrix(m);
	int i, j;
	for(i=0; i<m->rows; i++)
	{
		for(j=0; j<m->cols; j++)
		{
			sig = 1.0 / (1.0 + exp(-1.0 * matrix_get(copy, i, j)));
			matrix_set(copy, i, j, sig * (1-sig));
		}
	}
	return copy;
}

__device__ matrix_t* device_matrix_square(matrix_t* m)
{
	matrix_t* copy = copy_matrix(m);
	int i, j;
	for(i=0; i<m->rows; i++)
	{
		for(j=0; j<m->cols; j++)
		{
			matrix_set(copy, i, j, pow(matrix_get(copy, i, j), 2));
		}
	}
	return copy;
}

__device__ matrix_t* device_matrix_prepend_col(matrix_t* m, float value)
{
	matrix_t* result = matrix_constructor(m->rows, m->cols+1);
	unsigned int i, j;
	for(i=0; i<result->rows; i++)
	{
		matrix_set(result, i, 0, value);
	}
	for(i=0; i<m->rows; i++)
	{
		for(j=0; j<m->cols; j++)
		{
			matrix_set(result, i, j+1, matrix_get(m, i, j));
		}
	}
	return result;
}

__device__ matrix_t* device_matrix_remove_col(matrix_t* m)
{
	matrix_t* result = matrix_constructor(m->rows, m->cols-1);
	unsigned int i, j;
	for(i=0; i<result->rows; i++)
	{
		for(j=0; j<result->cols; j++)
		{
			matrix_set(result, i, j, matrix_get(m, i, j+1));
		}
	}
	return result;
}

__device__ matrix_t* device_matrix_prepend_row(matrix_t* m, float value)
{
	matrix_t* result = matrix_constructor(m->rows+1, m->cols);
	unsigned int i, j;
	for(i=0; i<result->cols; i++)
	{
		matrix_set(result, 0, i, value);
	}
	for(i=0; i<m->rows; i++)
	{
		for(j=0; j<m->cols; j++)
		{
			matrix_set(result, i+1, j, matrix_get(m, i, j));
		}
	}
	return result;
}

__device__ matrix_t* device_matrix_remove_row(matrix_t* m)
{
	matrix_t* result = matrix_constructor(m->rows-1, m->cols);
	unsigned int i, j;
	for(i=0; i<result->rows; i++)
	{
		for(j=0; j<result->cols; j++)
		{
			matrix_set(result, i, j, matrix_get(m, i+1, j));
		}
	}
	return result;
}

__device__ matrix_t* device_row_to_vector(matrix_t* m, unsigned int row)
{
	matrix_t* v = matrix_constructor(1, m->cols);
	unsigned int i;
	for(i=0; i<m->cols; i++)
	{
		vector_set(v, i, matrix_get(m, row, i));
	}
	return v;
}

__device__ matrix_t* device_col_to_vector(matrix_t* m, unsigned int col)
{
	matrix_t* v = matrix_constructor(1, m->rows);
	unsigned int i;
	for(i=0; i<m->rows; i++)
	{
		vector_set(v, i, matrix_get(m, i, col));
	}
	return v;
}

__device__ matrix_t* device_matrix_cell_multiply(matrix_t* m1, matrix_t* m2)
{
	assert(m1 != NULL && m2 != NULL);
	assert(m1->rows > 0 && m2->rows > 0 && m1->cols > 0 && m2->cols > 0);
	assert(m1->rows == m2->rows && m1->cols == m2->cols);

	matrix_t* product = matrix_constructor(m1->rows, m1->cols);

	int i, j;
	for(i=0; i<m1->rows; i++)
	{
		for(j=0; j<m1->cols; j++)
		{
			matrix_set(product, i, j, matrix_get(m1, i, j) * matrix_get(m2, i , j));
		}
	}
	return product;
}

__device__ float device_matrix_average(matrix_t* m)
{
	int i, j;
	float sum;
	for(i=0; i<m->rows; i++)
	{
		for(j=0; j<m->cols; j++)
		{
			sum += matrix_get(m, i, j);
		}
	}
	return sum / (m->rows * m->cols);
}

__device__ matrix_t* device_matrix_random(unsigned int rows, unsigned int cols, float range)
{
	srand(time(NULL));
	matrix_t *m = matrix_constructor(rows, cols);

	unsigned int i, j;
	for(i=0; i<rows; i++)
	{
		for(j = 0; j<cols; j++)
		{
			float random = ((float)(rand() % 1000)) / (float)1000;
			matrix_set(m, i, j, random * 2 * range - range);
		}
	}
	return m;
}

__device__ matrix_list_t* device_matrix_list_add(matrix_list_t* m1, matrix_list_t* m2)
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

 unsigned int device_matrix_memory_size(matrix_t* m)
{
	return sizeof(matrix_t) + sizeof(float) * rows + cols;
}

__device__ unsigned int device_matrix_list_memory_size(matrix_list_t* m)
{
	unsigned int memory_size = sizeof(matrix_list_t);
	unsigned int i;
	for(i=0; i<m->num; i++)
	{
		memory_size += device_matrix_memory_size(m->matrix_list[i]);
	}
	return memory_size;
}
