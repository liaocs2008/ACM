#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
  int key;
  Node *left, *right, *p;
} Node;

void INORDER_TREE_WALK(Node *x)
{
  if (x != NULL) {
    INORDER_TREE_WALK(x->left);
    printf("%d ", x->key);
    INORDER_TREE_WALK(x->right);
  }
}

Node* TREE_SEARCH(Node *x, int k)
{
  if ((NULL == x) || (k == x->key)) {
    return x;
  }
  if (k < x->key) {
    return TREE_SEARCH(x->left, k);
  } else {
    return TREE_SEARCH(x->right, k);
  }
}

Node* TREE_MINIMUM(Node *x)
{
  while (NULL != x->left)
    x = x->left;
  return x;
}

Node* TREE_MAXIMUM(Node *x)
{
  while (NULL != x->right)
    x = x->right;
  return x;
}

void TREE_INSERT(Node *&root, Node *z)
{
  Node *y=NULL, *x=root;
  while (NULL != x) {
    y = x;
    if (z->key < x->key) x = x->left;
    else x = x->right;
  }
  z->p = y;
  if (NULL == y) {
    root = z;
  } else {
    if (z->key < y->key) y->left = z;
    else y->right = z;
  }
}

Node* TREE_SUCCESSOR(Node *x)
{
  if (NULL != x->right)
    return TREE_MINIMUM(x->right);
  Node *y = x->p;
  while ((NULL != y) && (x == y->right)) {
    x = y;
    y = y->p;
  }
  return y;
}

Node* TREE_PREDECESSOR(Node *x)
{
  if (NULL != x->left)
    return TREE_MAXIMUM(x->left);
  Node *y = x->p;
  while ((NULL != y) && (x == y->left)) {
    x = y;
    y = y->p;
  }
  return y;
}

void TRANSPLANT(Node *&root, Node *u, Node *v)
{
  if (NULL == u->p) {
    root = v;
  } else if (u == u->p->left) {
    u->p->left = v;
  } else {
    u->p->right = v;
  }
  if (NULL != v) {
    v->p = u->p;
  }
}

void TREE_DELETE(Node *&root, Node *z)
{
  if (NULL == z->left) {
    TRANSPLANT(root, z, z->right);
  } else if (NULL == z->right) {
    TRANSPLANT(root, z, z->left);
  } else {
    // this is for the case that z has two children
    // so, z's successor y won't have left child, and it won't be empty
    Node *y = TREE_MINIMUM(z->right);
    // TO SPLICE Y OUT OF ITS LOCATION

    // if y lies with z's right subtree, not the root of this subtree,
    // replace y by its own and only child (right child)
    // if y happens to be the root, don't splice y out of its location
    if (y->p != z) {
      TRANSPLANT(root, y, y->right);
      y->right = z->right;
      y->right->p = y;
    }

    // replace z by y
    TRANSPLANT(root, z, y);
    y->left = z->left;
    y->left->p = y;
  }
}

int main()
{
  Node *root=NULL;

  Node a[10];
  int v[10]={2,3,4,1,6,7,5,8,0,9};
  for (int i=0; i<10; ++i) {
    a[i].key = v[i];
    a[i].left = NULL;
    a[i].right = NULL;
    a[i].p = NULL;
    TREE_INSERT(root, &a[i]);
  }

  printf("root=%d\n", root->key);
  INORDER_TREE_WALK(root);
  Node * tmp = TREE_MAXIMUM(root);
  printf("max=%d\n", tmp->key);
  tmp = TREE_MINIMUM(root);
  printf("min=%d\n", tmp->key);
  tmp = TREE_PREDECESSOR(root);
  printf("pre=%d\n", tmp->key);
  tmp = TREE_SUCCESSOR(root);
  printf("succ=%d\n", tmp->key);



  for (int i=0; i<10; ++i) {
    printf("[%d] del %d\n", i, a[i].key);
    TREE_DELETE(root, &a[i]);
    INORDER_TREE_WALK(root);
    if (NULL != root)
      printf(" root=%d\n", root->key);
  }


  return 0;
}
