#!/bin/python
V = 'uvwxyz'
E = {'u':'vx', 'v':'y', 'w':'yz', 'x':'v', 'y':'x', 'z':'z'}

def DFS(V, E):
  color = {v:0 for v in V}
  d = {v:0 for v in V}
  f = {v:0 for v in V}
  time = [0] # pass by reference
  for u in V:
    if color[u] == 0:
      print "FOREST"
      DFS_VISIT(V, E, u, color, d, f, time)
  return color, d, f, time

def DFS_VISIT(V, E, u, color, d, f, time):
  print u
  time[0] = time[0] + 1
  d[u] = time[0]
  color[u] = 1
  for v in E[u]:
    if color[v] == 0:
      DFS_VISIT(V, E, v, color, d, f, time)
  time[0] = time[0] + 1
  f[u] = time[0]
  color[u] = 2
  

if __name__ == "__main__":
  color,d,f,time = DFS(V, E)
  print color
  print d
  print f
  print time
