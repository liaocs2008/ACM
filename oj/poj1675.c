/*
   1. Assuming (0,0) to one berry (i.e, A) is picked as baseline, try to prove 
      the "No" case occurs only when angle BOA and angle COA are both within 
      the same sector.
   2. Angles can be positive and negative, but don't worry about sort and minus 
      operation.
*/

#include <stdio.h>
#include <math.h>
int main()
{
  int i, j, t, r, x[3], y[3], d[3], tmp;
  scanf("%d", &t);
  while (t > 0) {
    t = t - 1;
    scanf("%d %d %d %d %d %d %d", &r, &x[0], &y[0], &x[1], &y[1], &x[2], &y[2]);
    if (((x[0]==0)&&(y[0]==0))||((x[1]==0)&&(y[1]==0))||((x[2]==0)&&(y[2]==0))){
      printf("No\n");
      continue;
    } else {
      for (i=0; i<3; i+=1) {
        d[i] = 180.0 * atan2(y[i], x[i])/3.1415926;
        for (j=i; j>0; j-=1) {
          if (d[j] < d[j-1]) {
            tmp = d[j];
            d[j] = d[j-1];
            d[j-1] = tmp;
          }
        }
      }
      if ((d[1] - d[0] < 120.0) && (d[2] - d[0] < 120.0)) {
        printf("No\n");
      } else printf("Yes\n");
    }
  }
  return 0;
}
