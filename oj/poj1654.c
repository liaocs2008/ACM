/*
   1. Data type should be long long.
   2. Here remains a question: 
        why the direction arrays (X,Y) is not as decribed?
        I copied this from other's solution since this is the critical difference
        between theirs and mine.
        Originally I was coding like -
          {0, 0}, {-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {0, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1}
*/

#include <stdio.h>
#include <string.h>
int X[10] = {0,1,1,1,0,0,0,-1,-1,-1};
int Y[10] = {0,-1,0,1,-1,0,1,-1,0,1};  
int main()
{
  int i=0, j=0;
  int n=0, l=0, d=0;
  char input[1000010];
  long long area=0, x=0, y=0;

  scanf("%d", &n);
  for (i=0; i<n; i=i+1) {
    scanf("%s", input);
    l = strlen(input);
    if (l <= 3) {
      printf("0\n");
      continue;
    }
    area = 0;
    x = 0;
    y = 0;
    for (j=0; j<l-1; j=j+1) {
      d = input[j] - '0';
      area += (x+X[d])*y - x*(y+Y[d]);
      x += X[d];
      y += Y[d];
    }
    area = area >= 0 ? area : -area;
    if (area & 1)
      printf("%lld.5\n", area/2);
    else
      printf("%lld\n", area/2);
  }
  return 0;
}
