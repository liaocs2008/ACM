#!/bin/python
import sys
sys.path.insert(0, '../sort')
from min_heap import MIN_HEAP_INSERT, HEAP_DECREASE_KEY, HEAP_EXTRACT_MIN

V = 'abcdefghi'
W = {'ab':4, 'ah':8, 
     'ba':4, 'bc':8, 'bh':11, 
     'cb':8, 'ci':2, 'cd':7, 'cf':4, 
     'dc':7, 'df':14, 'de':9, 
     'ed':9, 'ef':10, 
     'fe':10, 'fd':14, 'fc':4, 'fg':2,
     'gf':2, 'gi':6, 'gh':1,
     'ha':8, 'hb':11, 'hg':1, 'hi':7,
     'ih':7, 'ic':2, 'ig':6}

E = {v:'' for v in V}
for uv in W.keys():
  u = uv[0]
  v = uv[1]
  E[u] = E[u] + v
#print E

import operator
def PRIM(V, E, W, s):
  Q = []
  key = {v:float('Inf') for v in V}
  pi = {v:'' for v in V}
  key[s] = 0
  for u in V:
    MIN_HEAP_INSERT(Q, (key[u], u), len(Q)) # (1,'a') > (0,'b'), although 'a' < 'b'
  MST = []
  while Q:
    u = HEAP_EXTRACT_MIN(Q, len(Q))[1]
    MST.append(u)
    for v in E[u]:
      if (key[v], v) in Q and W[u+v] < key[v]:
        pi[v] = u
        old_key = key[v]
        key[v] = W[u+v]
        HEAP_DECREASE_KEY(Q, Q.index((old_key, v)), (key[v], v))
  return MST

print PRIM(V, E, W, 'a')
