// gcc combine.c -O3 -DFLOAT -DPLUS
//
// I write 7 combine() functions according to the book
//
// other codes are from:
// [1] http://csapp.cs.cmu.edu/2e/code.html
// [2] https://www.cs.drexel.edu/~jjohnson/wi04/cs680/programs/opt/counter.c 

#include <stdio.h>
#include <stdlib.h>

#ifdef FLOAT
typedef float data_t;
#define DATA_NAME "Float"
#endif

#ifdef DOUBLE
typedef double data_t;
#define DATA_NAME "Double"
#endif

#ifdef EXTEND
typedef long double data_t;
#define DATA_NAME "Extended"
#endif

#ifdef INT
typedef int data_t;
#define DATA_NAME "Integer"
#endif

#ifdef LONG
typedef long data_t;
#define DATA_NAME "Long"
#endif

#ifdef CHAR
typedef char data_t;
#define DATA_NAME "Char"
#endif

#ifdef PROD
#define IDENT 1
#define OP  *
#define OP_NAME "Product"
#else
#ifdef DIV
#define OP /
#define IDENT 1
#define OP_NAME "Divide"
#else
#define IDENT 0
#define OP  +
#define OP_NAME "Sum"
#endif 
#endif 

static unsigned cyc_hi = 0;
static unsigned cyc_lo = 0;

/* Set *hi and *lo to the high and low order bits of the cycle counter.
   Implementation requires assembly code to use the rdtsc instruction. */

void access_counter(unsigned *hi, unsigned *lo)
{
     asm("rdtsc; movl %%edx, %0; movl %%eax, %1" /* read cycle counter */
         : "=r" (*hi), "=r" (*lo)                /* and move results to */
         : /* No input */                        /* the two outputs */
         : "%edx", "%eax");
}

/* Record the current value of the cycle counter. */
void start_counter()
{
     access_counter(&cyc_hi,&cyc_lo);
}

/* Return the number of cycles since the last call to start_counter. */

double get_counter()
{
     unsigned ncyc_hi, ncyc_lo;
     unsigned hi, lo, borrow;
     double result;

     /* get cycle counter */
     access_counter(&ncyc_hi, &ncyc_lo);

     /*  double precision subtraction */
     lo = ncyc_lo - cyc_lo;
     borrow = lo > ncyc_lo;
     hi = ncyc_hi - cyc_hi - borrow;
     result = (double) hi * (1 << 30) * 4 + lo;
     if (result < 0) {
        fprintf(stderr,"Error: counter returns neg value: %.0f\n",result);
     }
     return result;
}

typedef struct {
    long int len;
    data_t *data;
/* $end adt */
    long int allocated_len; /* NOTE: we don't use this field in the book */
/* $begin adt */ 
} vec_rec, *vec_ptr;

/* $begin vec */
/* Create vector of specified length */
vec_ptr new_vec(int len)
{
    /* allocate header structure */
    vec_ptr result = (vec_ptr) malloc(sizeof(vec_rec));
    if (!result)
        return NULL;  /* Couldn't allocate storage */
    result->len = len;
/* $end vec */
    /* We don't show this in the book */
    result->allocated_len = len;
/* $begin vec */
    /* Allocate array */
    if (len > 0) {
        data_t *data = (data_t *)calloc(len, sizeof(data_t));
	if (!data) {
	    free((void *) result);
 	    return NULL; /* Couldn't allocate storage */
	}
	result->data = data;
    }
    else
	result->data = NULL;
    return result;
}

/*
 * Retrieve vector element and store at dest.
 * Return 0 (out of bounds) or 1 (successful)
 */
int get_vec_element(vec_ptr v, int index, data_t *dest)
{
    if (index < 0 || index >= v->len)
	return 0;
    *dest = v->data[index];
    return 1;
}

/* Return length of vector */
long int vec_length(vec_ptr v)
{
    return v->len;
}
/* $end vec */


/* $begin get_vec_start */
data_t *get_vec_start(vec_ptr v)
{
    return v->data;
}
/* $end get_vec_start */


/*
 * Set vector element.
 * Return 0 (out of bounds) or 1 (successful)
 */
int set_vec_element(vec_ptr v, int index, data_t val)
{
    if (index < 0 || index >= v->len)
	return 0;
    v->data[index] = val;
    return 1;
}


/* Set vector length.  If >= allocated length, will reallocate */
void set_vec_length(vec_ptr v, int newlen)
{
    if (newlen > v->allocated_len) {
	free(v->data);
	v->data = (data_t*) calloc(newlen, sizeof(data_t));
	v->allocated_len = newlen;
    }
    v->len = newlen;
}


void combine1(vec_ptr v, data_t *dest)
{
  int i;

  *dest = IDENT;
  for (i = 0; i < vec_length(v); ++i) {
    data_t val;
    get_vec_element(v, i, &val);
    *dest = *dest OP val;
  }
}

void combine2(vec_ptr v, data_t *dest)
{
  int i;
  int length = vec_length(v);

  *dest = IDENT;
  for (i = 0; i < length; ++i) {
    data_t val;
    get_vec_element(v, i, &val);
    *dest = *dest OP val;
  }
}


void combine3(vec_ptr v, data_t *dest)
{
  int i;
  int length = vec_length(v);
  data_t *data = get_vec_start(v);

  *dest = IDENT;
  for (i = 0; i < length; ++i) {
    *dest = *dest OP data[i];
  }
}

void combine4(vec_ptr v, data_t *dest)
{
  int i;
  int length = vec_length(v);
  data_t *data = get_vec_start(v);
  data_t acc = IDENT;

  for (i = 0; i < length; ++i) {
    acc = acc OP data[i];
  }
  *dest = acc;
}

void combine5(vec_ptr v, data_t *dest)
{
  int i;
  int length = vec_length(v);
  int limit = length - 1;
  data_t *data = get_vec_start(v);
  data_t acc = IDENT;

  for (i = 0; i < limit; i += 2) {
    acc = (acc OP data[i]) OP data[i+1];
  }

  for (; i < length; ++i) {
    acc = acc OP data[i];
  }

  *dest = acc;
}

void combine6(vec_ptr v, data_t *dest)
{
  int i;
  int length = vec_length(v);
  int limit = length - 1;
  data_t *data = get_vec_start(v);
  data_t acc0 = IDENT;
  data_t acc1 = IDENT;

  for (i = 0; i < limit; i += 2) {
    acc0 = acc0 OP data[i];
    acc1 = acc1 OP data[i+1];
  }

  for (; i < length; ++i) {
    acc0 = acc0 OP data[i];
  }

  *dest = acc0 OP acc1;
}

void combine7(vec_ptr v, data_t *dest)
{
  int i;
  int length = vec_length(v);
  int limit = length - 1;
  data_t *data = get_vec_start(v);
  data_t acc = IDENT;

  for (i = 0; i < limit; i += 2) {
    acc = acc OP (data[i] OP data[i+1]);
  }

  for (; i < length; ++i) {
    acc = acc OP data[i];
  }

  *dest = acc;
}

int main(int argc, char * argv[])
{
  int n;
  double cycles;

  n = 1024;
  vec_ptr p = new_vec(n);
  data_t d;

  start_counter();
  combine1(p, &d);
  cycles = get_counter();
  printf("cycles for combine1 = %.0f\n",cycles/n);

  start_counter();
  combine2(p, &d);
  cycles = get_counter();
  printf("cycles for combine2 = %.0f\n",cycles/n);

  start_counter();
  combine3(p, &d);
  cycles = get_counter();
  printf("cycles for combine3 = %.0f\n",cycles/n);

  start_counter();
  combine4(p, &d);
  cycles = get_counter();
  printf("cycles for combine4 = %.0f\n",cycles/n);

  start_counter();
  combine5(p, &d);
  cycles = get_counter();
  printf("cycles for combine5 = %.0f\n",cycles/n);

  start_counter();
  combine6(p, &d);
  cycles = get_counter();
  printf("cycles for combine6 = %.0f\n",cycles/n);

  start_counter();
  combine7(p, &d);
  cycles = get_counter();
  printf("cycles for combine7 = %.0f\n",cycles/n);

  return 0;
}
