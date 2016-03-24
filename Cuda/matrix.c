#include "matrix.h"

matrix_t* matrix_constructor(unsigned int rows, unsigned int cols)
{
	assert(rows > 0 && cols > 0);

	matrix_t* m = (matrix_t*)malloc(sizeof(matrix_t) + sizeof(float) * rows * cols);
	
	assert(m != NULL);

	m->rows = rows;
	m->cols = cols;
	set_matrix(m, 0.0);

	return m;
}

float matrix_get(matrix_t* m, unsigned int x, unsigned int y)
{
	assert(m != NULL);
	assert(x >= 0 && x < m->rows && y >= 0 && y < m->cols);
	return (m->matrix[x * m->cols + y]);
}

void matrix_set(matrix_t* m, unsigned int x, unsigned int y, float value)
{
	assert(m != NULL);
	assert(x >= 0 && x < m->rows && y >= 0 && y < m->cols);
	m->matrix[x * m->cols + y] = value;
}

matrix_t* matrix_add(matrix_t* m1, matrix_t* m2)
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

matrix_t* matrix_subtract(matrix_t* m1, matrix_t* m2)
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

matrix_t* matrix_multiply(matrix_t* m1, matrix_t* m2)
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

matrix_t* matrix_scalar_multiply(matrix_t* m, float scalar)
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

matrix_t* matrix_transpose(matrix_t* m)
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


void set_matrix(matrix_t* m, float val)
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

void set_matrix_index(matrix_t* m)
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

matrix_t* copy_matrix(matrix_t* m)
{
	matrix_t* copy = matrix_constructor(m->rows, m->cols);
	memcpy(copy->matrix, m->matrix, sizeof(float)*m->rows*m->cols);
	return copy;
}

void print_matrix(matrix_t* m)
{
	int i, j;
	printf("%dx%d\n", m->rows, m->cols);

	for(i=0; i<m->rows; i++)
	{
		for(j=0; j<m->cols; j++)
		{
			printf("%f ", matrix_get(m, i, j));
		}
		printf("\n");
	}
}

matrix_list_t* matrix_list_constructor(unsigned int num)
{
	matrix_list_t* list = (matrix_list_t*)malloc(sizeof(matrix_list_t));
	list->num = num;
	list->matrix_list = (matrix_t**)malloc(sizeof(matrix_t*) * num);
	return list;
}

void free_matrix(matrix_t* m)
{
	assert(m != NULL);
	assert(m->matrix != NULL);
	free(m->matrix);
	free(m);
}

void free_matrix_list(matrix_list_t* m)
{
	assert(m != NULL);
	int i;
	for(i=0; i<m->num; i++)
	{
		free_matrix(m->matrix_list[i]);
	}
	free(m);
}

matrix_t* matrix_sigmoid(matrix_t* m)
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

matrix_t* matrix_sigmoid_gradient(matrix_t* m)
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

matrix_t* matrix_square(matrix_t* m)
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

matrix_t* matrix_prepend_col(matrix_t* m, float value)
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

matrix_t* matrix_remove_col(matrix_t* m)
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

matrix_t* matrix_prepend_row(matrix_t* m, float value)
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

matrix_t* matrix_remove_row(matrix_t* m)
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

matrix_t* row_to_vector(matrix_t* m, unsigned int row)
{
	matrix_t* v = matrix_constructor(1, m->cols);
	unsigned int i;
	for(i=0; i<m->cols; i++)
	{
		vector_set(v, i, matrix_get(m, row, i));
	}
	return v;
}

matrix_t* col_to_vector(matrix_t* m, unsigned int col)
{
	matrix_t* v = matrix_constructor(1, m->rows);
	unsigned int i;
	for(i=0; i<m->rows; i++)
	{
		vector_set(v, i, matrix_get(m, i, col));
	}
	return v;
}

matrix_t* matrix_cell_multiply(matrix_t* m1, matrix_t* m2)
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

matrix_t* load_from_file(const char* filename, unsigned int rows, unsigned int cols)
{
	matrix_t* m = matrix_constructor(rows, cols);

	char* line = NULL;
	size_t n = 0;
	FILE* stream = fopen(filename, "rb");

	int i, j;
	for(i=0; i<rows; i++)
	{
		int ret = getline(&line, &n, stream);
		assert(ret != EOF);

		char* tmp = strtok(line, ",");
		for(j=0; j<cols; j++)
		{
			assert(tmp != NULL);
			matrix_set(m, i, j, atof(tmp));
			tmp = strtok(NULL, ",");
		}
		n = 0;
		free(line);
	}
	return m;
}

float matrix_average(matrix_t* m)
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

void print_matrix_dimensions(matrix_t* m)
{
	printf("%dx%d\n", m->rows, m->cols);
}

matrix_t* matrix_random(unsigned int rows, unsigned int cols, float range)
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

unsigned int matrix_memory_size(matrix_t* m)
{
	return sizeof(matrix_t) + sizeof(float) * rows + cols;
}