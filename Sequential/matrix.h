#include "NNinclude.h"

// types
typedef struct matrix_t
{
	unsigned int rows;
	unsigned int cols;
	float matrix[];
} matrix_t;

typedef struct matrix_list_t
{
	unsigned int num;
	matrix_t** matrix_list;
} matrix_list_t;

typedef struct buffer_t
{
	unsigned long size;
	char* current_index;
	char pool[];
} buffer_t;

// constructors
matrix_t* matrix_constructor(unsigned rows, unsigned cols);
matrix_list_t* matrix_list_constructor(unsigned int num);

// arithmetic
matrix_t* matrix_add(matrix_t* m1, matrix_t* m2);
matrix_t* matrix_subtract(matrix_t* m1, matrix_t* m2);
matrix_t* matrix_multiply(matrix_t* m1, matrix_t* m2);
matrix_t* matrix_scalar_multiply(matrix_t* m, float scalar);
matrix_t* matrix_sigmoid(matrix_t* m);
matrix_t* matrix_square(matrix_t* m);
matrix_t* matrix_sigmoid_gradient(matrix_t* m);
matrix_t* matrix_cell_multiply(matrix_t* m1, matrix_t* m2);
matrix_t* matrix_transpose(matrix_t* m);

matrix_list_t* matrix_list_add(matrix_list_t* m1, matrix_list_t* m2);
matrix_list_t* matrix_list_subtract(matrix_list_t* m1, matrix_list_t* m2);
matrix_list_t* matrix_list_scalar_multiply(matrix_list_t* m1, float scalar);

// matrix generation
matrix_t* matrix_random(unsigned int rows, unsigned int cols, float range);
matrix_t* load_from_file(const char* filename, unsigned int rows, unsigned int cols);
matrix_t* copy_matrix(matrix_t* m);

// free
void free_matrix(matrix_t* m);
void free_matrix_list(matrix_list_t* m);

// get/set
float matrix_get(matrix_t* m, unsigned int x, unsigned int y);
void matrix_set(matrix_t* m, unsigned int x, unsigned int y, float value);
void set_matrix(matrix_t* m, float val);
void set_matrix_index(matrix_t* m);
float vector_get(matrix_t* v, unsigned int x);
void vector_set(matrix_t* v, unsigned int x, float value);

// print
void print_matrix(matrix_t* m);
void print_matrix_dimensions(matrix_t* m);

// average
float matrix_average(matrix_t* m);

// size
unsigned int matrix_memory_size(matrix_t* m);
unsigned int matrix_list_memory_size(matrix_list_t* m);

// matrix partitioning
matrix_list_t* matrix_list_add(matrix_list_t* m1, matrix_list_t* m2);
matrix_t* row_to_vector(matrix_t* m, unsigned int row);
matrix_t* col_to_vector(matrix_t* m, unsigned int col);
matrix_t* roll_matrix_list(matrix_list_t* list);
matrix_list_t* unroll_matrix_list(matrix_t* vector, int num, unsigned int sizes[][2]);
matrix_t* matrix_prepend_col(matrix_t* m, float value);
matrix_t* matrix_remove_col(matrix_t* m);
matrix_t* matrix_prepend_row(matrix_t* m, float value);
matrix_t* matrix_remove_row(matrix_t* m);

// Tests
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

unsigned int matrix_util_test();
unsigned int test_roll_unroll_matrices();

// manual profiling
void start_tracking();
void stop_tracking();
long get_memory_used();
long get_total_mallocs();

void set_buffer(buffer_t* b);
