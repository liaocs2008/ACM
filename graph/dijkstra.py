#!/bin/python
import sys
sys.path.insert(0, '../sort')
from min_heap import MIN_HEAP_INSERT, HEAP_DECREASE_KEY, HEAP_EXTRACT_MIN


V = 'stxyz'
W = {'st':10, 'sy':5,
     'tx':1, 'ty':2,
     'xz':4,
     'yz':2, 'yx':4, 'yt':3,
     'zx':6, 'zs':7}

E = {v:'' for v in V}
for u,v in W.keys():
  E[u] = E[u] + v
#print E


import operator
def DIJKSTRA(V, E, W, s):
  Q = []
  d = {v:float('Inf') for v in V}
  pi = {v:'' for v in V}
  d[s] = 0
  for u in V:
    MIN_HEAP_INSERT(Q, (d[u], u), len(Q)) # (1,'a') > (0,'b'), although 'a' < 'b'
  S = []
  while Q:
    u = HEAP_EXTRACT_MIN(Q, len(Q))[1]
    S.append(u)
    for v in E[u]:
      if d[u] + W[u+v] < d[v]:
        pi[v] = u
        old_d = d[v]
        d[v] = d[u] + W[u+v]
        HEAP_DECREASE_KEY(Q, Q.index((old_d, v)), (d[v], v))
  return S, pi, d

if __name__ == "__main__":
  S, pi, d = DIJKSTRA(V, E, W, 's')
  print S
  print pi
  print d
