#include "buffer.cuh"

__device__ buffer_t* buffer = NULL;

__device__ buffer_t* buffer_constructor(size_t size, void* memptr)
{
	buffer_t* buf = (buffer_t*)memptr;
	buf->size = size - sizeof(unsigned long) - sizeof(char*);
	buf->current_index = buf->pool;
	return buf;
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

__device__ buffer_t* get_buffer()
{
	return buffer;
}

__device__ void set_buffer(buffer_t* b)
{
	buffer = b;
}

__device__ void reset_buffer(buffer_t* b)
{
	b->current_index = b->pool;
}
