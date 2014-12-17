#!/bin/python
S = [1,2,4,1,5,8,9,11,13]
F = [3,5,7,8,9,10,11,14,16]

def SELECTOR(S, F, f):
  tmp = [0]
  for i in range(len(S)):
    if F[i] <= f:
      tmp.append(1+SELECTOR(S,F,S[i]))
  return max(tmp)

def SELECTOR_GREEDY(S, F, f):
  count = 0
  tmp = [-float('Inf') for i in range(len(F))]
  for i in range(len(F)):
    if F[i] <= f:
      # trying to find the nearest starting time
      tmp[i] = S[i]
      count = count + 1
  if not count:
    return 0
  index = tmp.index(max(tmp))
  return 1 + SELECTOR_GREEDY(S, F, S[index])

for finish in range(F[-1]+1):
  print "finish time set =", finish, SELECTOR(S,F,finish), SELECTOR_GREEDY(S,F,finish)
