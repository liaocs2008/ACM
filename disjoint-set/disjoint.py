#!/bin/python

# x[0] = p, x[1] = rank
def MAKE_SET(x):
  x[0] = x
  x[1] = 0

def FIND_SET(x):
  if x[0] != x:
    x[0] = FIND_SET(x[0])
  return x[0]

def LINK(x, y):
  if x[1] > y[1]:
    y[0] = x
  else:
    x[0] = y
    if x[1] == y[1]:
      y[1] += 1

def UNION(x, y):
  LINK(FIND_SET(x), FIND_SET(y))

if __name__ == "__main__":
  V = 'abcdefghij'
  E = {'a':'bc', 'b':'acd', 'c':'ab', 'd':'b', 'e':'fg', 'f':'e', 'g':'e',
       'h':'i', 'i':'h', 'j':''}
  def CONNECTED_COMPONENTS(V, E):
    V_1 = []
    for v in V:
      tmp = [0,0,v]
      MAKE_SET(tmp)
      V_1.append(tmp)
    for v in V:
      for u in E[v]:
        v_1 = [x for x in V_1 if x[2]==v][0]
        u_1 = [x for x in V_1 if x[2]==u][0]
        #it is dangerous to compare infinite list
        if FIND_SET(u_1)[2] != FIND_SET(v_1)[2]:
          UNION(u_1, v_1)
    return V_1
  tmp_v = CONNECTED_COMPONENTS(V, E)
  for v in tmp_v:
    print v
