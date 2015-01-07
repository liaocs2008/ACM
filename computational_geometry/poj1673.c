/*
   1. This formulation is derived by following equations (inner product):
        (x-x2)(x3-x1) + (y-y2)(y3-y1) = 0
        (x-x1)(x3-x2) + (y-y1)(y3-y2) = 0
   2. Don't bother to think about special cases. As seen in result:
        If y3-y1=0, then y2 != y3, y2 != y1, and x3 != x1.
        So the case (...) / 0 can't happen.
   3. 1e-8 is to adjust -0.0000 to 0.0000.
   4. To pass poj, only choose G++ compiler and print in the format of "%.4f" rather than "%.4lf".
   5. To prove, try to prove AO is perpendicular to BC. As a result O should be orthocenter.
      Extends N to P to let CN equal to PN, and then connect FP. Try to prove angle CAB equal to angle CFP.
      A little hint is that angle ACO plus angle FCN equals to 90 degrees.
*/

#include <stdio.h>

int main()
{
  int n=0;
  double x1,x2,x3,y1,y2,y3,hx,hy;
  scanf("%d", &n);
  while (n>0) {
    n = n - 1;
    scanf("%lf%lf%lf%lf%lf%lf", &x1, &y1, &x2, &y2, &x3, &y3);
    hy = 1e-8 + ( ( (x3-x1)*x2+(y3-y1)*y2 )*(x3-x2) - ( (x3-x2)*x1+(y3-y2)*y1 )*(x3-x1) )/( ( (y3-y1)*(x3-x2) - (y3-y2)*(x3-x1)) );
    hx = 1e-8 + ( ( (x3-x1)*x2+(y3-y1)*y2 )*(y3-y2) - ( (x3-x2)*x1+(y3-y2)*y1 )*(y3-y1) )/( ( (x3-x1)*(y3-y2) - (x3-x2)*(y3-y1)) );

    printf("%.4lf %.4lf\n", hx, hy);
  }
  return 0;
}
