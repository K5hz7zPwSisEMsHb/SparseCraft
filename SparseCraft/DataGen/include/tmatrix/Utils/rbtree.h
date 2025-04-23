#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tmatrix/common.h>

typedef enum { RED, BLACK } Color;

typedef struct RBTreeNode {
    int key;
    bit256 val;
    Color color;
    struct RBTreeNode *left, *right, *parent;
} RBTreeNode;

typedef struct RBTree {
    RBTreeNode *root;
    RBTreeNode *nil; // 哨兵节点
} RBTree;

RBTree *create_rbtree() {
    RBTree *tree = (RBTree *)malloc(sizeof(RBTree));
    tree->nil = (RBTreeNode *)malloc(sizeof(RBTreeNode));
    tree->nil->color = BLACK;
    tree->root = tree->nil;
    return tree;
}

RBTreeNode *create_node(RBTree *tree, int key, bit256 val) {
    RBTreeNode *node = (RBTreeNode *)malloc(sizeof(RBTreeNode));
    node->key = key;
    node->val = val;
    node->color = RED; // 新插入的节点总是红色
    node->left = tree->nil;
    node->right = tree->nil;
    node->parent = tree->nil;
    return node;
}

void left_rotate(RBTree *tree, RBTreeNode *x) {
    RBTreeNode *y = x->right;
    x->right = y->left;
    if (y->left != tree->nil) {
        y->left->parent = x;
    }
    y->parent = x->parent;
    if (x->parent == tree->nil) {
        tree->root = y;
    } else if (x == x->parent->left) {
        x->parent->left = y;
    } else {
        x->parent->right = y;
    }
    y->left = x;
    x->parent = y;
}

void right_rotate(RBTree *tree, RBTreeNode *y) {
    RBTreeNode *x = y->left;
    y->left = x->right;
    if (x->right != tree->nil) {
        x->right->parent = y;
    }
    x->parent = y->parent;
    if (y->parent == tree->nil) {
        tree->root = x;
    } else if (y == y->parent->right) {
        y->parent->right = x;
    } else {
        y->parent->left = x;
    }
    x->right = y;
    y->parent = x;
}

void rb_insert_fixup(RBTree *tree, RBTreeNode *z) {
    while (z->parent->color == RED) {
        if (z->parent == z->parent->parent->left) {
            RBTreeNode *y = z->parent->parent->right;
            if (y->color == RED) {
                z->parent->color = BLACK;
                y->color = BLACK;
                z->parent->parent->color = RED;
                z = z->parent->parent;
            } else {
                if (z == z->parent->right) {
                    z = z->parent;
                    left_rotate(tree, z);
                }
                z->parent->color = BLACK;
                z->parent->parent->color = RED;
                right_rotate(tree, z->parent->parent);
            }
        } else {
            RBTreeNode *y = z->parent->parent->left;
            if (y->color == RED) {
                z->parent->color = BLACK;
                y->color = BLACK;
                z->parent->parent->color = RED;
                z = z->parent->parent;
            } else {
                if (z == z->parent->left) {
                    z = z->parent;
                    right_rotate(tree, z);
                }
                z->parent->color = BLACK;
                z->parent->parent->color = RED;
                left_rotate(tree, z->parent->parent);
            }
        }
    }
    tree->root->color = BLACK;
}

void rb_insert(RBTree *tree, int key, bit256 val) {
    RBTreeNode *z = create_node(tree, key, val);
    RBTreeNode *y = tree->nil;
    RBTreeNode *x = tree->root;

    while (x != tree->nil) {
        y = x;
        if (z->key < x->key) {
            x = x->left;
        } else {
            x = x->right;
        }
    }

    z->parent = y;
    if (y == tree->nil) {
        tree->root = z;
    } else if (z->key < y->key) {
        y->left = z;
    } else {
        y->right = z;
    }

    rb_insert_fixup(tree, z);
}

RBTreeNode *rb_search(RBTree *tree, int key) {
    RBTreeNode *current = tree->root;

    while (current != tree->nil) {
        if (key == current->key) {
            return current; // 找到节点
        } else if (key < current->key) {
            current = current->left;
        } else {
            current = current->right;
        }
    }

    return NULL; // 未找到
}

int rb_update(RBTree *tree, int key, bit256 new_val, bit256 (*op)(bit256, bit256)) {
    RBTreeNode *node = rb_search(tree, key);
    if (node == NULL) {
        rb_insert(tree, key, new_val);
        return 0; // 未找到指定 key 的节点，更新失败
    }

    // 更新 val 值
    node->val = op(node->val, new_val);
    return 1; // 更新成功
}

typedef struct RBTreeIterator {
    RBTreeNode *current;
    RBTree *tree;
} RBTreeIterator;

RBTreeIterator *rb_iterator_init(RBTree *tree) {
    RBTreeIterator *iter = (RBTreeIterator *)malloc(sizeof(RBTreeIterator));
    iter->tree = tree;
    iter->current = tree->root;

    // 寻找最小节点
    if (iter->current != tree->nil) {
        while (iter->current->left != tree->nil) {
            iter->current = iter->current->left;
        }
    }

    return iter;
}

int rb_iterator_has_next(RBTreeIterator *iter) {
    return iter->current != iter->tree->nil;
}

RBTreeNode *rb_iterator_next(RBTreeIterator *iter) {
    if (iter->current == iter->tree->nil) {
        return NULL;
    }

    RBTreeNode *node = iter->current;

    // 如果有右子树，那么下一个节点是右子树的最小节点
    if (iter->current->right != iter->tree->nil) {
        iter->current = iter->current->right;
        while (iter->current->left != iter->tree->nil) {
            iter->current = iter->current->left;
        }
    } else {
        // 否则，向上回溯到第一个左链接的祖先节点
        while (iter->current->parent != iter->tree->nil && iter->current == iter->current->parent->right) {
            iter->current = iter->current->parent;
        }
        iter->current = iter->current->parent;
    }

    return node;
}

void rb_free_node(RBTree *tree, RBTreeNode *node) {
    if (node != tree->nil) {
        rb_free_node(tree, node->left);
        rb_free_node(tree, node->right);
        free(node);
    }
}


void rb_free(RBTree *tree) {
    if (tree != NULL) {
        rb_free_node(tree, tree->root);
        free(tree->nil); // 释放哨兵节点
        free(tree);      // 释放红黑树结构
    }
}
