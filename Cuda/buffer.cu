#include "buffer.cuh"

__device__ buffer_t* buffer_constructor(size_t size, void* memptr)
{
	buffer_t* buffer = (buffer_t*)memptr;
	buffer->size = size - sizeof(unsigned long) - sizeof(char*);
	buffer->current_index = buffer->pool;
	return buffer;
}
__device__ void* buffer_malloc(buffer_t* buffer, size_t size)
{
	if(size > buffer->size - (buffer->current_index - buffer->pool))
	{
		return NULL;
	}

	void* ptr = buffer->current_index;
	buffer->current_index += size;
	return ptr;
}

__device__ void free_buffer(buffer_t* buffer)
{
	//free(buffer);
}

__device__ void reset_buffer(buffer_t* buffer)
{
	buffer->current_index = buffer->pool;
}
