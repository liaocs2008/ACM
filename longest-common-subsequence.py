#!/bin/python
x = 'abcbdab'
y = 'bdcaba'

def LCS_LENGTH_2(x, y):
  # c is with size (|x|+1)*(|y|+1)
  rows = len(x)+1
  cols = len(y)+1
  c = [[0 for j in range(cols)] for i in range(rows)]
  for i in range(1,rows):
    for j in range(1,cols):
      if x[i-1] == y[j-1]: c[i][j] = 1 + c[i-1][j-1]
      elif c[i-1][j] >= c[i][j-1]: c[i][j] = c[i-1][j]
      else: c[i][j] = c[i][j-1]
  return c[-1][-1], c
length, c = LCS_LENGTH_2(x, y)
print length
for r in c:
  print r

def PRINT_LCS(c, x, y):
  output = []
  i = len(x)
  j = len(y)
  while i > 0 and j > 0:
    if c[i][j] == 1+c[i-1][j-1]:
      output.insert(0, x[i-1])
      i = i-1
      j = j-1
    elif c[i-1][j] >= c[i][j-1]:
      i = i-1
    else:
      j = j-1
  return output

print PRINT_LCS(c, x, y)

def LCS_LENGTH(x, y):
  if not x or not y:
    return 0
  if x[-1] == y[-1]:
    return 1 + LCS_LENGTH(x[0:-1], y[0:-1])
  else:
    return max(LCS_LENGTH(x[0:-1], y), LCS_LENGTH(x, y[0:-1]))
print LCS_LENGTH(x, y)
