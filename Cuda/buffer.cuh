#include "NNinclude.cuh"

typedef struct buffer_t
{
	unsigned long size;
	char* current_index;
	char pool[];
} buffer_t;

__device__ buffer_t* buffer_constructor(size_t size, void* memptr);
__device__ void* buffer_malloc(buffer_t* buffer, size_t size);
__device__ void free_buffer(buffer_t* buffer);
__device__ buffer_t* get_buffer();
__device__ void set_buffer(buffer_t* b);
__device__ void reset_buffer(buffer_t* b);
