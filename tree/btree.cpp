#include <stdio.h>
#include <stdlib.h>

#define T 4

typedef struct node {
  int num;
  bool leaf;
  // c0 <= k1 <= c1 <= ... <= c(2t-2) <= k(2t-1) <= c(2t-1)
  int keys[2 * T]; // sentinel
  node* children[2 * T];
} node;

typedef struct position {
  node *x; 
  int pos;
} position;

position search(node *x, int k)
{
  position p = {.x = NULL, .pos = -1};

  int i = 1;
  while ((i <= x->num) && (k > x->keys[i])) { i += 1;}

  // keys[i-1] < k <= keys[i]
  if ((i <= x->num) && (k == x->keys[i])) { // node found
    p.x = x;
    p.pos = k;
    return p;
  } else if (true == x->leaf) {
    return p;
  } else {
    // k < keys[i]
    // load children from disk if necessary
    return search(x->children[i - 1], k);
  }
}

void split(node *x, int i, node *y) // x->children[i], i.e. y, is full, y->num = 2T-1
{
  int j = 0;

  // initial new node z
  node *z = (node *) calloc(1, sizeof(node));
  z->leaf = y->leaf;
  z->num = T - 1;

  // copy right part (T-1) of y to new node z
  for (j = 1; j < T; j += 1) 
    z->keys[i] = y->keys[j + T];
  if (false == y->leaf) {
    for (j = 1; j < T; j += 1) 
      z->children[i] = y->children[j + T];
  }
  y->num = T - 1; // note that the middle one is passed to parent

  // insert new node z to x
  // c(j-1) <= k(j) <= c(j) ---> c(j) <= k(j+1) <= c(j+1)
  for (j = x->num; j >= i; j -= 1) { // j for key index
    x->keys[j + 1] = x->keys[j];
    x->children[j + 1] = x->children[j];
  }
  
  // k(i) and c(i) are moved
  x->children[i] = z;
  x->keys[i] = y->keys[T];
  x->num += 1;
  
  // write y, z, x to disk if necessary
}

void insert_nonfull(node *x, int k)
{
  int i = x->num;
  if (true == x->leaf) {
    while ((i >= 1) && (k < x->keys[i])) {
      x->keys[i + 1] = x->keys[i];
      i -= 1;
    }
    x->keys[i + 1] = k;
    x->num += 1;
    // write x to disk if necessary
  } else {
    while ((i >= 1) && (k < x->keys[i])) {
      i -= 1;
    }
    i += 1; // after this step, keys[i-1] < k < keys[i]
    // read x->children[i] if necessary

    if (2 * T - 1 == x->children[i]->num) { // full found
      split(x, i, x->children[i]);
      if (k > x->keys[i]) i += 1;
    }

    insert_nonfull(x->children[i], k);
  }
}

void insert(node *&root, int k)
{
  node *r = root;
  if (2 * T - 1 == r->num) {
    node *s = (node *) calloc(1, sizeof(node));
    root = s;
    s->leaf = false;
    s->num = 0;
    s->children[1] = r;
    split(s, 1, r);
    insert_nonfull(s, k);
  } else insert_nonfull(r, k);
}



int main()
{

  return 0;
}
