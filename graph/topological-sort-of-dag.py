#!/bin/python
from DFS import DFS
V = 'cbadefgh'
E = {'a':'b', 'b':'efc', 'c':'gd', 'd':'ch', 'e':'af', 'f':'g', 'g':'fh', 'h':'h'}

def TOPOLOGICAL_SORT(V, E):
  print "==================="
  color,d,f,time = DFS(V, E)
  E_transpose = {v:'' for v in V}
  for v in V:
    for u in E[v]:
      E_transpose[u] = E_transpose[u] + v
  V_sorted = [v for (f,v) in sorted([(f[v],v) for v in V], reverse=True)]
  print "==================="
  DFS(V_sorted, E_transpose)

TOPOLOGICAL_SORT(V, E)
