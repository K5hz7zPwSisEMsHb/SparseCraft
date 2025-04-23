#include <tmatrix/Utils/msg.h>
#include <tmatrix/Utils/MemoryPool.h>

MemoryPool *memory_pool_init(size_t size, size_t alignment)
{
    MemoryPool *mp = (MemoryPool *)malloc(sizeof(MemoryPool));
    if (!mp)
    {
        echo(error, "Failed to allocate memory pool");
        exit(EXIT_FAILURE);
    }

    // 分配内存池并确保对齐
    if (posix_memalign((void**)&mp->pool, alignment, size) != 0)
    {
        echo(error, "Failed to allocate aligned memory pool");
        free(mp);
        exit(EXIT_FAILURE);
    }

    echo(debug, "Memory Pool Range: %p ~ %p", mp->pool, mp->pool + size);

    mp->pool_size = size;
    mp->_offset = 0;
    mp->_alignment = alignment;
    return mp;
}

u_int64_t memory_pool_alloc(MemoryPool *mp, size_t size)
{
    u_int64_t res = mp->_offset;
    mp->_offset += size;
    return res;
}

void memory_pool_free(MemoryPool *mp)
{
    if (mp)
    {
        free(mp->pool);
        free(mp);
    }
}