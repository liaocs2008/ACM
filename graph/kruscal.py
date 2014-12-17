#!/bin/python
import sys
sys.path.insert(0, '../disjoint-set')
from disjoint import MAKE_SET, FIND_SET, UNION

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
def KRUSCAL(V, W):
  V_1 = []
  for v in V:
    tmp = [0,0,v]
    MAKE_SET(tmp)
    V_1.append(tmp)
  E = sorted(W.items(), key=operator.itemgetter(1))
  MST = []
  for i in E:
    u = i[0][0]
    v = i[0][1]
    u_1 = [x for x in V_1 if x[2]==u][0]
    v_1 = [x for x in V_1 if x[2]==v][0]
    if FIND_SET(u_1)[2] != FIND_SET(v_1)[2]:
      MST.append(i)
      UNION(u_1, v_1)
  return MST

if __name__ == "__main__":
  print KRUSCAL(V,W)
