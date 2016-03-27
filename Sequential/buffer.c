#include "buffer.h"

static buffer_t* buffer = NULL;

buffer_t* buffer_constructor(size_t size)
{
	buffer_t* buf = (buffer_t*)malloc(sizeof(buffer_t) + size);
	buf->current_index = buf->pool;
	buf->size = size;
	return buf;
}
void* buffer_malloc(buffer_t* buffer, size_t size)
{
	if(size > buffer->size - (buffer->current_index - buffer->pool))
	{
		return NULL;
	}

	void* ptr = buffer->current_index;
	buffer->current_index += size;
	return ptr;
}

void free_buffer(buffer_t* buffer)
{
	free(buffer);
}

buffer_t* get_buffer()
{
	return buffer;
}

void set_buffer(buffer_t* b)
{
	buffer = b;
}

void reset_buffer(buffer_t* b)
{
	b->current_index = b->pool;
}
