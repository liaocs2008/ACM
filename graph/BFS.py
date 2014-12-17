#!/bin/python
V = 'rstuvwxy'
E = {'r':'sv', 's':'rw', 't':'uwx', 'u':'txy', 'v':'r', 'w':'stx', 'x':'tuwy', 'y':'xu'}

def BFS(V, E, s):
  color = {v:0 for v in V}
  d = {v:-1 for v in V}
  pi = {v:'' for v in V}
  color[s] = 1
  d[s] = 0
  pi[s] = ''
  Q = []
  Q.append(s)
  while Q:
    u = Q.pop(0)
    for v in E[u]:
      if color[v] == 0:
        Q.append(v)
        color[v] = 1
        d[v] = d[u]+1
        pi[v] = u
    color[u] = 2
    print u
  print color
  print d
  print pi

BFS(V, E, 's')
