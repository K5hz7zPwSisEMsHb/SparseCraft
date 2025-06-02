#include <tmatrix/Utils/rbtree.h>

RBTree *create_rbtree(copy_func_t copy_func, free_func_t free_func, merge_func_t merge_func) {
    RBTree *tree = (RBTree *)malloc(sizeof(RBTree));
    tree->nil = (RBTreeNode *)malloc(sizeof(RBTreeNode));
    tree->nil->color = BLACK;
    tree->root = tree->nil;
    tree->size = 0;

    tree->copy_func = copy_func;
    tree->free_func = free_func;
    tree->merge_func = merge_func;
    return tree;
}

RBTreeNode *create_node(RBTree *tree, int key, void*val) {
    RBTreeNode *node = (RBTreeNode *)malloc(sizeof(RBTreeNode));
    node->key = key;
    node->val = tree->copy_func(val); // 使用复制函数复制 val
    node->color = RED; // 新插入的节点总是红色
    node->left = tree->nil;
    node->right = tree->nil;
    node->parent = tree->nil;
    tree->size++;
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

void rb_insert(RBTree *tree, int key, void*val) {
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

    return NULL;
}

int rb_update(RBTree *tree, int key, void* new_val) {
    RBTreeNode *node = rb_search(tree, key);
    if (node == NULL) {
        rb_insert(tree, key, new_val);
        return 1;
    }
    tree->merge_func(node->val, new_val);
    return 1;
}

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
        tree->free_func(node->val);
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
