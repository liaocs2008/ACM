#include <stdio.h>

typedef enum {red, black} color;

typedef struct node node;
struct node {
  int key;
  node *p, *left, *right;
  color c;
};

// sentinel
node sentinel = {.key = 0, .p = &sentinel, .left = &sentinel, .right = &sentinel, .c = black};
node *nil = &sentinel;

void inorder_walk(node *x)
{
  if (nil != x) {
    inorder_walk(x->left);
    printf("%d, ", x->key);
    inorder_walk(x->right);
  }
}

void left_rotate(node *& root, node *x) // this functon may update root
{
  node *y = x->right;
  // subtree belta
  x->right = y->left;
  if (nil != y->left) y->left->p = x;
  // x's parent
  y->p = x->p;
  if (nil == x->p) root = y;
  else {
    if (x->p->left == x) x->p->left = y;
    else x->p->right = y;
  }
  // y 
  y->left = x;
  // x
  x->p = y;
}

void right_rotate(node *&root, node *y) // this functon may update root
{
  node *x = y->left;
  // subtree belta
  y->left = x->right;
  if (nil != x->right) x->right->p = y;
  // y's parent
  x->p = y->p;
  if (nil == y->p) root = x;
  else {
    if (y->p->left == x) y->p->left = x;
    else y->p->right = x;
  }
  // x 
  x->right = y;
  // y
  y->p = x;
}

void insert_fixup(node *&root, node *z) // this functon won't update root, but left_rotate or right_rotate may
{
  node *y = nil; 
  // this must be interesting to notice:
  // if z is only node, i.e., the root, its parent is nil whose color is black
  // if z is a child of root, its parent's (root) color is black
  // so, here comes the conclusion, z must has 2 levels above itself
  while (red == z->p->c) {
    if (z->p == z->p->p->left) {
      y = z->p->right; // y is the uncle of z
      if (red == y->c) {
        y->c = black;
        z->p->c = black;
        z->p->p->c = red; // this may cause the same problem as z was, red-red
        z = z->p->p;
      } else { // y has black color
        if (z == z->p->right) { // convert z to its parent's left child
          z = z->p;
          left_rotate(root, z);
        }
        z->p->c = black;
        z->p->p->c = red;
        right_rotate(root, z->p->p);
      }
    } else { // following are symmetric
      y = z->p->p->left;
      if (red == y->c) {
        y->c = black;
        z->p->c = black;
        z->p->p->c = red;
        z = z->p->p;
      } else {
        if (z == z->p->left) {
          z = z->p;
          right_rotate(root, z);
        }
        z->p->c = black;
        z->p->p->c = red;
        left_rotate(root, z->p->p);
      }
    }
  }
  root->c = black;
}

void insert(node *&root, node *z) // this function may update root
{
  node *y = nil, *x = root;
  // find position to insert, y is always x's parent
  while (nil != x) {
    y = x;
    if (z->key < x->key) x = x->left;
    else x = x->right;
  }
  // insert
  if (nil == y) root = z;
  else {
    if (z->key < y->key) y->left = z;
    else y->right = z;
  }
  // set up z
  z->p = y;
  z->left = nil;
  z->right = nil;
  z->c = red;
  insert_fixup(root, z);
}

int main()
{
  int i = 0;
  node a[10], *root = nil;
  for (i = 0; i < 9; i += 1) {
    a[i].key = i;
    insert(root, &a[i]);
  }
  inorder_walk(root);

  return 0;
}
