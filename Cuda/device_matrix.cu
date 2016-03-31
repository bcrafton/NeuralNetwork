#include "matrix.cuh"

__device__ matrix_t* device_matrix_constructor(buffer_t* buffer, unsigned int rows, unsigned int cols)
{
	//assert(rows > 0 && cols > 0);
	
	matrix_t* m = (matrix_t*)buffer_malloc(buffer, sizeof(matrix_t) + sizeof(float) * rows * cols);

	m->rows = rows;
	m->cols = cols;
	device_set_matrix(m, 0.0);

	return m;
}

__device__ matrix_t* device_matrix_add(buffer_t* buffer, matrix_t* m1, matrix_t* m2)
{
	//assert(m1 != NULL && m2 != NULL);
	//assert(m1->rows > 0 && m2->rows > 0 && m1->cols > 0 && m2->cols > 0);
	//assert(m1->rows == m2->rows && m1->cols == m2->cols);

	matrix_t* sum = device_matrix_constructor(buffer, m1->rows, m1->cols);

	int i, j;
	for(i=0; i<m1->rows; i++)
	{
		for(j=0; j<m1->cols; j++)
		{
			device_matrix_set(sum, i, j, device_matrix_get(m1, i, j) + device_matrix_get(m2, i, j));
		}
	}
	return sum;
}

__device__ matrix_t* device_matrix_subtract(buffer_t* buffer, matrix_t* m1, matrix_t* m2)
{
	//assert(m1 != NULL && m2 != NULL);
	//assert(m1->rows > 0 && m2->rows > 0 && m1->cols > 0 && m2->cols > 0);
	//assert(m1->rows == m2->rows && m1->cols == m2->cols);

	matrix_t* difference = device_matrix_constructor(buffer, m1->rows, m1->cols);

	int i, j;
	for(i=0; i<m1->rows; i++)
	{
		for(j=0; j<m1->cols; j++)
		{
			device_matrix_set(difference, i, j, device_matrix_get(m1, i, j) - device_matrix_get(m2, i , j));
		}
	}
	return difference;
}

__device__ matrix_t* device_matrix_multiply(buffer_t* buffer, matrix_t* m1, matrix_t* m2)
{
	if(!(m1->rows > 0 && m2->rows > 0 && m1->cols > 0 && m2->cols > 0))
	{
		//printf("%d %d %d %d", m1->rows, m2->rows, m1->cols, m2->cols);
	}
	//assert(m1 != NULL && m2 != NULL);
	//assert(m1->rows > 0 && m2->rows > 0 && m1->cols > 0 && m2->cols > 0);
	//assert(m1->cols == m2->rows);

	matrix_t* product = device_matrix_constructor(buffer, m1->rows, m2->cols);

	int i, j, k;
	for(i=0; i<product->rows; i++)
	{
		for(j=0; j<product->cols; j++)
		{
			for(k=0; k<m1->cols; k++)
			{
				device_matrix_set(product, i, j, device_matrix_get(product, i, j) + device_matrix_get(m1, i, k) * device_matrix_get(m2, k, j));
			}
		}
	}
	return product;
}

__device__ matrix_t* device_matrix_scalar_multiply(buffer_t* buffer, matrix_t* m, float scalar)
{
	//assert(m!= NULL);
	//assert(m->rows > 0 && m->cols > 0);

	matrix_t* product = device_matrix_constructor(buffer, m->rows, m->cols);

	int i, j;
	for(i=0; i<m->rows; i++)
	{
		for(j=0; j<m->cols; j++)
		{
			device_matrix_set(product, i, j, device_matrix_get(m, i, j) * scalar);
		}
	}
	return product;
}

__device__ matrix_t* device_matrix_sigmoid(buffer_t* buffer, matrix_t* m)
{
	matrix_t* copy = device_copy_matrix(buffer, m);
	int i, j;
	for(i=0; i<m->rows; i++)
	{
		for(j=0; j<m->cols; j++)
		{
			device_matrix_set(copy, i, j, 1.0 / (1.0 + exp(-1.0 * device_matrix_get(copy, i, j))));
		}
	}
	return copy;
}

__device__ matrix_t* device_matrix_sigmoid_gradient(buffer_t* buffer, matrix_t* m)
{
	float sig;
	matrix_t* copy = device_copy_matrix(buffer, m);
	int i, j;
	for(i=0; i<m->rows; i++)
	{
		for(j=0; j<m->cols; j++)
		{
			sig = 1.0 / (1.0 + exp(-1.0 * device_matrix_get(copy, i, j)));
			device_matrix_set(copy, i, j, sig * (1-sig));
		}
	}
	return copy;
}

__device__ matrix_t* device_matrix_square(buffer_t* buffer, matrix_t* m)
{
	matrix_t* copy = device_copy_matrix(buffer, m);
	int i, j;
	for(i=0; i<m->rows; i++)
	{
		for(j=0; j<m->cols; j++)
		{
			device_matrix_set(copy, i, j, pow(device_matrix_get(copy, i, j), 2));
		}
	}
	return copy;
}

__device__ matrix_t* device_matrix_cell_multiply(buffer_t* buffer, matrix_t* m1, matrix_t* m2)
{
	//assert(m1 != NULL && m2 != NULL);
	//assert(m1->rows > 0 && m2->rows > 0 && m1->cols > 0 && m2->cols > 0);
	//assert(m1->rows == m2->rows && m1->cols == m2->cols);

	matrix_t* product = device_matrix_constructor(buffer, m1->rows, m1->cols);

	int i, j;
	for(i=0; i<m1->rows; i++)
	{
		for(j=0; j<m1->cols; j++)
		{
			device_matrix_set(product, i, j, device_matrix_get(m1, i, j) * device_matrix_get(m2, i , j));
		}
	}
	return product;
}

__device__ matrix_t* device_matrix_transpose(buffer_t* buffer, matrix_t* m)
{
	//assert(m!= NULL);
	//assert(m->rows > 0 && m->cols > 0);

	matrix_t* transpose = device_copy_matrix(buffer, m);
	transpose->rows = m->cols;
	transpose->cols = m->rows;
	int i, j;
	for(i=0; i<m->rows; i++)
	{
		for(j=0; j<m->cols; j++)
		{
			device_matrix_set(transpose, j, i, device_matrix_get(m, i, j));
		}
	}
	return transpose;
}

__device__ matrix_t* device_copy_matrix(buffer_t* buffer, matrix_t* m)
{
	matrix_t* copy = device_matrix_constructor(buffer, m->rows, m->cols);
	memcpy(copy->matrix, m->matrix, sizeof(float)*m->rows*m->cols);
	return copy;
}

__device__ void device_free_matrix(matrix_t* m)
{
}

__device__ float device_matrix_get(matrix_t* m, unsigned int x, unsigned int y)
{
	//assert(m != NULL);
	////assert(x >= 0 && x < m->rows && y >= 0 && y < m->cols);
	return (m->matrix[x * m->cols + y]);
}

__device__ void device_matrix_set(matrix_t* m, unsigned int x, unsigned int y, float value)
{
	//assert(m != NULL);
	////assert(x >= 0 && x < m->rows && y >= 0 && y < m->cols);
	m->matrix[x * m->cols + y] = value;
}

__device__ void device_set_matrix(matrix_t* m, float val)
{
	//assert(m != NULL);
	//assert(m->rows > 0 && m->cols > 0);

	int i, j;
	for(i=0; i<m->rows; i++)
	{
		for(j=0; j<m->cols; j++)
		{
			device_matrix_set(m, i, j, val);
		}
	}
}

__device__ void device_set_matrix_index(matrix_t* m)
{
	//assert(m != NULL);
	//assert(m->rows > 0 && m->cols > 0);

	int i, j;
	for(i=0; i<m->rows; i++)
	{
		for(j=0; j<m->cols; j++)
		{
			device_matrix_set(m, i, j, i * m->cols + j);
		}
	}
}

__device__ unsigned int device_matrix_memory_size(matrix_t* m)
{
	return sizeof(matrix_t) + sizeof(float) * m->rows * m->cols;
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

__device__ matrix_t* device_row_to_vector(buffer_t* buffer, matrix_t* m, unsigned int row)
{
	matrix_t* v = device_matrix_constructor(buffer, 1, m->cols);
	unsigned int i;
	for(i=0; i<m->cols; i++)
	{
		device_vector_set(v, i, device_matrix_get(m, row, i));
	}
	return v;
}

__device__ matrix_t* device_col_to_vector(buffer_t* buffer, matrix_t* m, unsigned int col)
{
	matrix_t* v = device_matrix_constructor(buffer, 1, m->rows);
	unsigned int i;
	for(i=0; i<m->rows; i++)
	{
		device_vector_set(v, i, device_matrix_get(m, i, col));
	}
	return v;
}

__device__ matrix_t* device_matrix_prepend_col(buffer_t* buffer, matrix_t* m, float value)
{
	matrix_t* result = device_matrix_constructor(buffer, m->rows, m->cols+1);
	unsigned int i, j;
	for(i=0; i<result->rows; i++)
	{
		device_matrix_set(result, i, 0, value);
	}
	for(i=0; i<m->rows; i++)
	{
		for(j=0; j<m->cols; j++)
		{
			device_matrix_set(result, i, j+1, device_matrix_get(m, i, j));
		}
	}
	return result;
}

__device__ matrix_t* device_matrix_remove_col(buffer_t* buffer, matrix_t* m)
{
	matrix_t* result = device_matrix_constructor(buffer, m->rows, m->cols-1);
	unsigned int i, j;
	for(i=0; i<result->rows; i++)
	{
		for(j=0; j<result->cols; j++)
		{
			device_matrix_set(result, i, j, device_matrix_get(m, i, j+1));
		}
	}
	return result;
}

__device__ matrix_t* device_matrix_prepend_row(buffer_t* buffer, matrix_t* m, float value)
{
	matrix_t* result = device_matrix_constructor(buffer, m->rows+1, m->cols);
	unsigned int i, j;
	for(i=0; i<result->cols; i++)
	{
		device_matrix_set(result, 0, i, value);
	}
	for(i=0; i<m->rows; i++)
	{
		for(j=0; j<m->cols; j++)
		{
			device_matrix_set(result, i+1, j, device_matrix_get(m, i, j));
		}
	}
	return result;
}

__device__ matrix_t* device_matrix_remove_row(buffer_t* buffer, matrix_t* m)
{
	matrix_t* result = device_matrix_constructor(buffer, m->rows-1, m->cols);
	unsigned int i, j;
	for(i=0; i<result->rows; i++)
	{
		for(j=0; j<result->cols; j++)
		{
			device_matrix_set(result, i, j, device_matrix_get(m, i+1, j));
		}
	}
	return result;
}

__device__ void device_matrix_add_to(matrix_t* m1, matrix_t* m2)
{
	int i, j;
	for(i=0; i<m1->rows; i++)
	{
		for(j=0; j<m1->cols; j++)
		{
			device_matrix_set(m1, i, j, device_matrix_get(m1, i, j) + device_matrix_get(m2, i, j));
		}
	}
}
