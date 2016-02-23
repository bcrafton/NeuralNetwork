#include "NNinclude.h"

typedef struct matrix_t
{
	unsigned int rows;
	unsigned int cols;
	double* matrix;
} matrix_t;

typedef struct matrix_list_t
{
	unsigned int num;
	matrix_t** matrix_list;
} matrix_list_t;

matrix_t* matrix_constructor(unsigned rows, unsigned cols);
matrix_list_t* matrix_list_constructor(unsigned int num);
matrix_t* matrix_add(matrix_t* m1, matrix_t* m2);
matrix_t* matrix_subtract(matrix_t* m1, matrix_t* m2);
matrix_t* matrix_multiply(matrix_t* m1, matrix_t* m2);
matrix_t* matrix_scalar_multiply(matrix_t* m, double scalar);
matrix_t* matrix_transpose(matrix_t* m);
matrix_t* copy_matrix(matrix_t* m);
void free_matrix(matrix_t* m);
matrix_t* matrix_sigmoid(matrix_t* m);
matrix_t* matrix_square(matrix_t* m);
matrix_t* matrix_sigmoid_gradient(matrix_t* m);
matrix_t* matrix_cell_multiply(matrix_t* m1, matrix_t* m2);

matrix_t* load_from_file(char* filename, unsigned int rows, unsigned int cols);

void set_matrix(matrix_t* m, double val);
void set_matrix_index(matrix_t* m);
inline double matrix_get(matrix_t* m, unsigned int x, unsigned int y);
inline void matrix_set(matrix_t* m, unsigned int x, unsigned int y, double value);

void print_matrix(matrix_t* m);
void print_matrix_dimensions(matrix_t* m);

unsigned int matrix_test();
unsigned int test_matrix_add();
unsigned int test_matrix_set();
unsigned int test_set_matrix_index();
unsigned int test_set_matrix();
unsigned int test_matrix_subtract();
unsigned int test_matrix_scalar_multiply();
unsigned int test_matrix_multiply();
unsigned int test_matrix_transpose();
unsigned int test_matrix_prepend_col();
unsigned int test_matrix_remove_col();
unsigned int test_matrix_sigmoid();
unsigned int test_matrix_square();
unsigned int test_row_to_vector();
unsigned int test_col_to_vector();
unsigned int test_matrix_load();
unsigned int test_matrix_prepend_row();
unsigned int test_matrix_remove_row();
unsigned int test_matrix_random();

matrix_t* roll_matrix_list(matrix_list_t* list);
matrix_list_t* unroll_matrix_list(matrix_t* vector, int num, unsigned int sizes[][2]);
matrix_t* matrix_prepend_col(matrix_t* m, double value);
matrix_t* matrix_remove_col(matrix_t* m);
matrix_t* matrix_prepend_row(matrix_t* m, double value);
matrix_t* matrix_remove_row(matrix_t* m);

unsigned int matrix_util_test();
unsigned int test_roll_unroll_matrices();

double vector_get(matrix_t* v, unsigned int x);
void vector_set(matrix_t* v, unsigned int x, double value);

matrix_t* row_to_vector(matrix_t* m, unsigned int row);
matrix_t* col_to_vector(matrix_t* m, unsigned int col);

double matrix_average(matrix_t* m);

matrix_t* matrix_random(unsigned int rows, unsigned int cols, double range);
