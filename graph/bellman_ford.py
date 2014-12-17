#/bin/python
V = 'stxyz'
W = {'st':6, 'sy':7,
     'tx':5, 'ty':8,
     'xt':-2,
     'yx':-3, 'yz':9,
     'zx':7, 'zs':2}

def BELLMAN_FORD(V,W,s):
  d = {v:float('Inf') for v in V}
  d[s] = 0
  pi = {v:'' for v in V}
  for i in range(len(V)-1):
    for e in W.keys():
      u,v = e
      if d[v] > d[u] + W[e]:
        d[v] = d[u] + W[e]
        pi[v] = u
  b = True
  for e in W.keys():
    u,v = e
    if d[u] > d[v] + W[e]:
      b = False
      break
  return d, pi, b

if __name__ == "__main__":
  d,pi,b = BELLMAN_FORD(V, W, 's')
  print b
  print d
  print pi
