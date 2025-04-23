#pragma once
#include <tmatrix/Utils/msg.h>
#include <string.h>
#include <tmatrix/common.h>

typedef struct
{
    MatIndex *key;
    MatValue *val;
} hash_key_val_t;

typedef struct
{
    MatIndex   *key;
    bit256 *val;
} hash_key_bitmap_t;

unsigned int nextPow2(unsigned int size)
{
    size--;
    size |= size >> 1;
    size |= size >> 2;
    size |= size >> 4;
    size |= size >> 8;
    size |= size >> 16;
    size++;
    return size;
}

hash_key_val_t *create_key_val_hash(unsigned int *expect_size)
{
    *expect_size = nextPow2(*expect_size);
    hash_key_val_t *hash = (hash_key_val_t *)malloc(sizeof(hash_key_val_t));
    hash->key = (MatIndex *)malloc(sizeof(MatIndex) * (*expect_size));
    hash->val = (MatValue *)malloc(sizeof(MatValue) * (*expect_size));

    memset(hash->key, -1, sizeof(MatIndex) * (*expect_size));
    return hash;
}

hash_key_bitmap_t *create_key_bitmap_hash(unsigned int expect_size)
{
    hash_key_bitmap_t *hash = (hash_key_bitmap_t*)malloc(sizeof(hash_key_bitmap_t));
    hash->key = (MatIndex *)malloc(sizeof(MatIndex) * (expect_size));
    hash->val = (bit256*) calloc(expect_size, sizeof(bit256));
    memset(hash->key, -1, sizeof(MatIndex) * (expect_size));
    return hash;
}

void insert_key_val_hash(hash_key_val_t *hash, unsigned int size, MatIndex key, MatValue val)
{
    const unsigned int _mask = size - 1;
    unsigned int index = key & _mask;
    while (hash->key[index] != -1 && hash->key[index] != key)
    {
        index = (index + 1) & _mask;
    }
    if (hash->key[index] == key)
        hash->val[index] += val;
    else
    {
        hash->key[index] = key;
        hash->val[index] = val;
    }
}

void _insert_key_bitmap_hash(MatIndex* keys, bit256* vals, unsigned int size, MatIndex key, bit256 val)
{
    const unsigned int _mask = size - 1;
    unsigned int index = key & _mask;
    while (keys[index] != -1 && keys[index] != key)
    {
        index = (index + 1) & _mask;
    }
    if (keys[index] == key){
        vals[index] |= val;
    }
    else
    {
        keys[index] = key;
        vals[index] = val;
    }
}

void insert_key_bitmap_hash(hash_key_bitmap_t *hash, unsigned int size, MatIndex key, bit256 val)
{
    _insert_key_bitmap_hash(hash->key, hash->val, size, key, val);
}

void iter_key_val_hash(hash_key_val_t *hash, unsigned int size, MatIndex *col_ptr, MatValue *val_ptr)
{
    for (unsigned int i = 0; i < size; i++)
    {
        if (hash->key[i] != -1)
        {
            *col_ptr++ = hash->key[i];
            *val_ptr++ = hash->val[i];
        }
    }
}

void iter_key_bitmap_hash(
    // input
    hash_key_bitmap_t *hash, unsigned int size, 
    // output
    MatIndex *col_ptr, bit256 *val_ptr
)
{
    for (unsigned int i = 0; i < size; i++)
    {
        if (hash->key[i] != -1 && hash->val[i].any())
        {
            *col_ptr++ = hash->key[i];
            *val_ptr++ = hash->val[i];
        }
    }
}

int count_key_bitmap_hash(hash_key_bitmap_t *hash, unsigned int size)
{
    int count = 0;
    for (unsigned int i = 0; i < size; i++)
    {
        if (hash->key[i] != -1 && hash->val[i].any())
        {
            count++;
        }
    }
    return count;
}

u_int32_t gather_and_resize_hash(hash_key_bitmap_t **table, unsigned int expect_size)
{
    u_int32_t idx = 0;
    for (u_int32_t i = 0; i < expect_size; i++)
    {
        if ((*table)->key[i] != -1)
        {
            if (idx != i)
            {
                (*table)->key[idx] = (*table)->key[i];
                (*table)->val[idx] = (*table)->val[i];
            }
            ++idx;
        }
    }
    // *table = (hash_key_bitmap_t *)realloc(*table, sizeof(hash_key_bitmap_t) + sizeof(MatIndex) * (idx) + sizeof(bit256) * (idx));
    (*table)->key = (MatIndex *)realloc((*table)->key, sizeof(MatIndex) * (idx));
    (*table)->val = (bit256 *)realloc((*table)->val, sizeof(bit256) * (idx));
    return idx;
}

#define free_hash(hash) \
    free(hash->key);            \
    free(hash->val);            \
    free(hash);
