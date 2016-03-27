#include "NNinclude.h"

typedef struct buffer_t
{
	unsigned long size;
	char* current_index;
	char pool[];
} buffer_t;

buffer_t* buffer_constructor(size_t size);
void* buffer_malloc(buffer_t* buffer, size_t size);
void free_buffer(buffer_t* buffer);
buffer_t* get_buffer();
void set_buffer(buffer_t* b);
void reset_buffer(buffer_t* b);
