/*
   This won't pass poj unfortunately. I haven't figured out the reason. But it 
   works well on data provided by others.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct POINT {double x1, x2;} POINT;
int cmp(const void* p1, const void* p2) 
{
  return ((POINT*)p1)->x1 - ((POINT*)p2)->x1 > 0 ? 1 : -1;
}

int main()
{
  POINT *a=NULL;
  double x1, x2, y, x3, x4, y1, p[3], x5, x6, y2;
  double l, tmp, f;
  int i, n;
  while( scanf("%lf %lf %lf", &x1, &x2, &y) ) {
    if ((x1 == 0) && (x2 == 0) && (y == 0)) break;
    else {
      scanf("%lf %lf %lf", &p[0], &p[1], &p[2]);
      y2 = p[2];
      
      scanf("%d", &n);
      a = (POINT*)calloc(n+1, sizeof(POINT));
      
      for (i = 0; i < n; i += 1) {
        scanf("%lf %lf %lf", &x3, &x4, &y1);
        if ((y1 > y) || (y1 < y2)) { x5 = p[0]; x6 = p[0]; }
        else if (y1 == y2) { x5 = x3; x6 = x4; }
        else if (y1 == y) {x5 = p[0]; x6 = p[1];}
        else {
          x5 = ((y2 - y1) * x2 + (y - y2) * x3)/(y - y1);
          x6 = ((y - y2) * x4 - (y1 - y2) * x1)/(y - y1);
        }
        a[i].x1 = x5 <= p[0] ? p[0] : (x5 > p[1] ? p[1] : x5);
        a[i].x2 = x6 >= p[1] ? p[1] : (x6 < p[0] ? p[0] : x6);
      }
      a[n].x1 = p[1];

      qsort(a, n, sizeof(POINT), cmp);

      f = a[0].x2;
      l = a[0].x1 - p[0];
      for (i = 1; i <= n; i += 1) {
        if (a[i].x1 <= f) {
          if (f < a[i].x2) f = a[i].x2;
        }
        else {
          tmp = a[i].x1 - f;
          f = a[i].x2;
          if (tmp > l) l = tmp;
        } 
      }

      if (fabs(l) < 1e-8) printf("No View\n");
      else printf("%.2lf\n", l);
      free(a);
    }
  }
  return 0;
}
