#include <stdio.h>
#include <math.h>
int skew(float a, float b)
{
  int n = (b >= 1) + (b - 1) / 0.8660254;
  return floor(a) * n - (a - floor(a) < 0.5) * n / 2;
}
int main()
{
  int g = 0, s = 0, s1 = 0, s2 = 0;
  float a = 0, b = 0;
  while( scanf("%f %f", &a, &b) != EOF ) {
    g = floor(a) * floor(b);
    s1 = skew(a, b);
    s2 = skew(b, a);
    s = s1 >= s2 ? s1 : s2;
    if (g >= s) printf("%d grid\n", g);
    else printf("%d skew\n", s);
  }
  return 0;
}
