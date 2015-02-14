/*
OUTPUT:
   insert No.0 node, 0,
   insert No.1 node, 0, 1,
   insert No.2 node, 0, 1, 2,
   insert No.3 node, 0, 1, 2, 3,
   insert No.4 node, 0, 1, 2, 3, 4,
   insert No.5 node, 0, 1, 2, 3, 4, 5,
   insert No.6 node, 0, 1, 2, 3, 4, 5, 6,
   insert No.7 node, 0, 1, 2, 3, 4, 5, 6, 7,
   insert No.8 node, 0, 1, 2, 3, 4, 5, 6, 7, 8,
   insert No.9 node, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
   remove No.0 node, 1, 2, 3, 4, 5, 6, 7, 8, 9,
   remove No.1 node, 2, 3, 4, 5, 6, 7, 8, 9,
   remove No.2 node, 3, 4, 5, 6, 7, 8, 9,
   remove No.3 node, 4, 5, 6, 7, 8, 9,
   remove No.4 node, 5, 6, 7, 8, 9,
   remove No.5 node, 6, 7, 8, 9,
   remove No.6 node, 7, 8, 9,
   remove No.7 node, 8, 9,
   remove No.8 node, 9,
   remove No.9 node,
 */

#include <stdio.h>
#include <assert.h>

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

node* minimum(node *x)
{
  while (nil != x->left) x = x->left;
  return x;
}

node* successor(node *x)
{
  if (nil != x->right) return minimum(x->right);
  else {
    node *y = x->p;
    while ((nil != y) && (x == y->right)) {
      x = y;
      y = y->p;
    }
    return y;
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
  // if z is the only node, i.e., the root, its parent is nil whose color is black
  // if z is a child of root, its parent's (root) color is black
  // so, here comes the conclusion: if it goes into the loop, z must has 2 levels above itself
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

void remove_fixup(node *&root, node *&x) // this function may change root and x both
{
  node *w = nil;
  while ((root != x) && (black == x->c)) {
    // assume x with an extra black color
    // we are going to remove the extra black color
    if (x == x->p->left) {
      w = x->p->right;
      if (red == w->c) { // convert this case to other case
        w->c = black;
        x->p->c = red;
        left_rotate(root, x->p);
        w = x->p->right;
      }

      if ((black == w->left->c) && (black == w->right->c)) { //case 2
        w->c = red;
        x = x->p; // after x changes to its parent, its color will be reb, successfully finish
      } else if (black == w->right->c) { //case 3
        w->left->c = black;
        w->c = red;
        right_rotate(root, w);
        w = x->p->right;
      } else { //case 4
        w->c = x->p->c; //red
        x->p->c = black;
        w->right->c = black;
        left_rotate(root, x->p);
        x = root; // successfully finish
      }
    } else {
      // following are symmetric
      w = x->p->left;
      if (red == w->c) { 
        w->c = black;
        x->p->c = red;
        right_rotate(root, x->p);
        w = x->p->left;
      }

      if ((black == w->left->c) && (black == w->right->c)) { 
        w->c = red;
        x = x->p; 
      } else if (black == w->left->c) { 
        w->right->c = black;
        w->c = red;
        left_rotate(root, w);
        w = x->p->left;
      } else { 
        w->c = x->p->c; 
        x->p->c = black;
        w->left->c = black;
        right_rotate(root, x->p);
        x = root; 
      }
    }
  }
}

void remove(node *& root, node *z) // this function may update root
{
  node *y = nil, *x = nil;
  // y, either successor of z or z
  if ((nil == z->left) || (nil == z->right)) y = z; // this case, just remove z between its parent and its children
  else y = successor(z);
  // x, a child of y 
  if (nil != y->left) x = y->left;
  else x = y->right;
  x->p = y->p; // WARNING: this may change sentinel's parent, handled in fixup function
  // fix y's parent
  if (nil == y->p) root = x;
  else {
    if (y == y->p->left) y->p->left = x;
    else y->p->right = x;
  }
  // remove z, by now, y is already removed from the tree
  if (y != z) z->key = y->key; // here, also move satellite data if existing
  if (black == y->c) remove_fixup(root, x);
}

int main()
{
  int i = 0;
  node a[10], *root = nil;
  for (i = 0; i < 10; i += 1) {
    a[i].key = i;
    insert(root, &a[i]);
    printf("insert No.%d node, ", i);
    inorder_walk(root);
    printf("\n");
  }

  for (i = 9; i > 0; i -= 1) {
    remove(root, &a[i]);
    printf("remove No.%d node, ", i);
    inorder_walk(root);
    printf("\n");
  }

  // check if sentinel is changed
  assert(nil->key == 0);
  assert(nil->p == &sentinel);
  assert(nil->left == &sentinel);
  assert(nil->right == &sentinel);
  assert(nil->c == black);

  return 0;
}
