/*
   This can't pass poj. But I think I have tried it in the right way. Strange.
 */
#include <stdio.h>
#include <math.h>
#define distance(x1, y1, x2, y2) sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
int n = 0, c[101], x[101][41], y[101][41];
double t[101][41];
int lookup(int i, int j)
{
  int k = 0;
  double d = 0;
  if (t[i][j] >= 0) return t[i][j];
  else {
    t[i][j] = lookup(i - 1, 0) + distance(x[i - 1][0], y[i - 1][0], x[i][j], y[i][j]);
    for (k = 1; k < c[i - 1]; k += 1) {
      d = lookup(i - 1, k) + distance(x[i - 1][k], y[i - 1][k], x[i][j], y[i][j]);
      if (t[i][j] > d) t[i][j] = d;
    } 
    return t[i][j];
  }
}
int main()
{
  int i = 0, j = 0, k = 0, s = 0;
  double d = -1, tmp = 0;
  scanf("%d", &n);
  for (i = 0; i < n; i += 1) {
    scanf("%d", &c[i]);
    for (j = 0; j < c[i]; j += 1) {
      scanf("%d %d", &x[i][j], &y[i][j]);
      t[i][j] = -1;
    }
  }
  for (i = 0; i < c[0]; i += 1) t[0][i] = 0;
  for (j = 0; j < c[0]; j += 1) {
    for (i = 0; i < c[n - 1]; i += 1) {
      tmp = lookup(n - 1, i) + distance(x[n-1][i], y[n-1][i], x[0][j], y[0][j]);
      if ((d < 0) || (d > tmp)) d = tmp;
    }
  }
  printf("%d\n", (int)(100 * d));
  return 0;
}
