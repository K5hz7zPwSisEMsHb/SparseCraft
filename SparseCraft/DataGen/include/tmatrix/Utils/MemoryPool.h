#pragma once

#include <stdlib.h>
#include <stdio.h>

struct MemoryPool
{
    char *pool;
    size_t pool_size;
    size_t _offset;
    size_t _alignment;
};

MemoryPool *memory_pool_init(size_t size, size_t alignment);
u_int64_t memory_pool_alloc(MemoryPool *mp, size_t size);
void memory_pool_free(MemoryPool *mp);