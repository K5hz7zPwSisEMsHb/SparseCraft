#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tmatrix/common.h>

typedef enum { RED, BLACK } Color;

typedef struct RBTreeNode {
    int key;
    // dense_tile val;
    void*val;
    Color color;
    struct RBTreeNode *left, *right, *parent;
} RBTreeNode;

typedef void* (*copy_func_t)(const void*);
typedef void (*free_func_t)(void*);
typedef void (*merge_func_t)(void*, const void*);

typedef struct RBTree {
    RBTreeNode *root;
    RBTreeNode *nil; // 哨兵节点
    int size;

    copy_func_t copy_func; // 用于复制节点值的函数
    free_func_t free_func; // 用于释放节点值的函数
    merge_func_t merge_func; // 用于合并节点值的函数
} RBTree;

RBTree *create_rbtree(copy_func_t copy_func, free_func_t free_func, merge_func_t merge_func);
void rb_insert(RBTree *tree, int key, void *val);
RBTreeNode *rb_search(RBTree *tree, int key);
int rb_update(RBTree *tree, int key, void *new_val);

typedef struct RBTreeIterator {
    RBTreeNode *current;
    RBTree *tree;
} RBTreeIterator;

RBTreeIterator *rb_iterator_init(RBTree *tree);
int rb_iterator_has_next(RBTreeIterator *iter);
RBTreeNode *rb_iterator_next(RBTreeIterator *iter);
void rb_free_node(RBTree *tree, RBTreeNode *node);
void rb_free(RBTree *tree);
