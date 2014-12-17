#!/bin/python
P = [1,5,8,9]
L = [1,2,3,4]
n = 20

Q = [-1] * n
def MEMORIZED_CUT_ROD(P, L, n):
  tmp = [0]
  for i in range(len(L)):
    if L[i] <= n:
      if -1 == Q[n-L[i]]:
        Q[n-L[i]] = MEMORIZED_CUT_ROD(P, L, n-L[i])
      tmp.append(P[i] + Q[n-L[i]])
  return max(tmp)
print MEMORIZED_CUT_ROD(P, L, n)
#print Q

def MEMORIZED_CUT_ROD_2(P, L, n):
  Q = [-1] * (n+1)
  for l in range(n+1):
    tmp = [0]
    for i in range(len(L)):
      if L[i] <= l:
        tmp.append(P[i] + Q[l-L[i]])
    Q[l] = max(tmp)
    #print Q[l]
  return Q[n]
print MEMORIZED_CUT_ROD_2(P, L, n)

def CUT_ROD(P, L, n):
  tmp = [0]
  for i in range(len(L)):
    if L[i] <= n:
      tmp.append(P[i]+CUT_ROD(P, L, n-L[i]))
  return max(tmp)
print CUT_ROD(P,L,n)
