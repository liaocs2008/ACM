#!/bin/python
def MERGE_SORT(a, p, r):
  if p < r:
    q = (p+r)/2
    MERGE_SORT(a, p, q)
    MERGE_SORT(a, q+1, r)
    MERGE(a, p, q, r)

def MERGE(a, p, q, r):
  n1 = q - p + 1
  L = [0]*n1
  for i in range(n1):
    L[i] = a[p+i]
  n2 = r - q
  R = [0]*n2
  for j in range(n2):
    R[j] = a[1+q+j] # L has been responsible for q, no need to care q in R
  # Set sentinel
  L.append(float('Inf'))
  R.append(float('Inf'))
  i = 0
  j = 0
  for k in range(p, r+1): # scope is p..r, including r
    if L[i] <= R[j]:
      a[k] = L[i]
      i = i+1
    else:
      a[k] = R[j]
      j = j+1

a = [1,3,9,8,0,2,5,6,7,4]
MERGE_SORT(a, 0, len(a)-1)
print a
