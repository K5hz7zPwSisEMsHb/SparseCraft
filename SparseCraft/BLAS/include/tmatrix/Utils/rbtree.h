#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tmatrix/common.h>

typedef enum { RED, BLACK } Color;

typedef struct RBTreeNode {
    int key;
    dense_tile val;
    Color color;
    struct RBTreeNode *left, *right, *parent;
} RBTreeNode;

typedef struct RBTree {
    RBTreeNode *root;
    RBTreeNode *nil; // 哨兵节点
    int size;
} RBTree;

RBTree *create_rbtree();
void rb_insert(RBTree *tree, int key, dense_tile&val);
RBTreeNode *rb_search(RBTree *tree, int key);
int rb_update(RBTree *tree, int key, dense_tile&new_val, dense_tile&(*op)(dense_tile&, dense_tile&));

typedef struct RBTreeIterator {
    RBTreeNode *current;
    RBTree *tree;
} RBTreeIterator;

RBTreeIterator *rb_iterator_init(RBTree *tree);
int rb_iterator_has_next(RBTreeIterator *iter);
RBTreeNode *rb_iterator_next(RBTreeIterator *iter);
void rb_free_node(RBTree *tree, RBTreeNode *node);
void rb_free(RBTree *tree);
