#include <stdio.h>
#include <string.h>
typedef struct {int x, y;} point;
int cross_product(point p0, point p1, point p2)
{
  return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}
int main()
{
  int m = 0, n = 0, x1 = 0, y1 = 0, x2 = 0, y2 = 0, B[5001];
  point U[5001], L[5001], p[5001];
  while (scanf("%d", &n)) {
    if (0 == n) break;
    int i = 0, l = 0, r = n - 1, mid = 0;
    scanf("%d %d %d %d %d", &m, &x1, &y1, &x2, &y2);
    memset((void *) B, 0, 5001 * sizeof(int));
    for (i = 0; i < n; i += 1) {
      scanf("%d %d", &U[i].x, &L[i].x);
      U[i].y = y1;
      L[i].y = y2;
    }
    U[n].x = L[n].x = x2;
    U[n].y = y1;
    L[n].y = y2;

    for (i = 0; i < m; i += 1) {
      scanf("%d %d", &p[i].x, &p[i].y);
      printf("%d, %d\n", p[i].x, p[i].y);
      i = 0, l = 0, r = n - 1, mid = 0;
      while (l < r) {
        mid = (l + r) / 2;
        printf("l=%d, r=%d, mid=%d\n", l, r, mid);
        if (cross_product(p[i], U[mid], L[mid]) > 0) l = mid + 1;
        else r = mid;
      }
      if (cross_product(p[l], U[l], L[l]) > 0) {
        printf("positive, l+1=%d\n", l + 1);
        B[l + 1] += 1;
      }
      else {
        printf("negative, l=%d\n", l);
        B[l] += 1;
      }
    }
    for (i = 0; i <= n; i += 1) printf("%d: %d\n", i, B[i]);
  }
  return 0;
}
