/* OUTPUT
   pattern:        C       G       T       A       T       G       C       G
   text has C
   replace A of text to G
   text has T
   text has A
   text has T
   insert G to text
   text has C
   text has G
 */

/*
   PATTERN CGTATGCG
   TEXT    CATATCG 
   "replace A of text to G", CATATCG -> CGTATCG
   "insert G to text", CGTATCG -> CGTATGCG (this becomes equal to PATTERN)
 */

#include <stdio.h>

#define N 1024
int table[N][N] = {0};

void dp(char *p, int lp, char *t, int lt)
{
  lp++;lt++;
  int i = 0, j = 0;
  for (i = 0; i < lp; ++i) table[i][0] = i;
  for (i = 0; i < lt; ++i) table[0][i] = 0;
  for (i = 1; i < lp; ++i)
  {
    for (j = 1; j < lt; ++j)
    {
      table[i][j] = table[i-1][j-1] + (p[i-1] == t[j-1] ? 0 : 1);
      if (table[i][j] > table[i][j-1] + 1) table[i][j] = table[i][j-1] + 1;
      if (table[i][j] > table[i-1][j] + 1) table[i][j] = table[i-1][j] + 1;
    }
  }

  int min = 1e6;
  for (i = 0; i < lt; ++i)
  {
    if (table[lp-1][i] < min)
    {
      min = table[lp-1][i];
      j = i;
    }
  }

  printf("pattern:\t");
  for (i = lp-1; i > 0; --i)
    printf("%c\t", p[i-1]);
  printf("\n");

  for (i = lp-1; i > 0 && j > 0;)
  {
    if (table[i][j] == table[i-1][j] + 1)
    {
      printf("insert %c to text\n", p[i-1]);
      i = i - 1;
    }
    else if (table[i][j] == table[i][j-1] + 1)
    {
      printf("remove %c of text\n", t[j-1]);
      j = j - 1;
    }
    else
    {
      if (t[j-1] != p[i-1])
        printf("replace %c of text to %c\n", t[j-1], p[i-1]);
      else
        printf("text has %c\n", t[j-1]);
      i = i - 1;
      j = j - 1;
    }
  }
}

int main()
{
  dp("GCGTATGC", 8, "TATTGGCTATACGGTT", 16); 
  return 0;
}
