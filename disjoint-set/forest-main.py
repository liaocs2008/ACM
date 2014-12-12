class Vertex:
  p=None
  rank=0
  name=""

def MAKE_SET(x):
  x.p = x
  x.rank = 0

def UNION(x, y): 
  return LINK(FIND_SET(x), FIND_SET(y))

def LINK(x, y):
  if x.rank > y.rank:
    y.p = x
  else:
    x.p = y
    if x.rank == y.rank:
      y.rank = y.rank + 1

def FIND_SET(x):
  if x != x.p:
    x.p = FIND_SET(x.p)
  return x.p

tmp_v = "abcdefghij"
tmp_e = {"a":"bc", "b":"acd", "c":"a", "d":"b", 
         "e":"fg", "f":"e", "g":"e", "h":"i",
         "i":"h", "j":""}
V=[]
E=[]
class Edge:
  l = None
  r = None

for n in tmp_v:
  # Vertices
  v = Vertex()
  v.name = n
  V.append(v)
  print v.name

for v in V:
  # Edges
  for m in tmp_e[v.name]:
    e = Edge()
    e.l = v
    for t in V:
      if t.name == m:
        e.r = t
        print e.l.name, e.r.name
        break
    E.append(e)

def TEST(V, E):
  for v in V:
    MAKE_SET(v)
  for e in E:
    if FIND_SET(e.l) != FIND_SET(e.r):
      #print FIND_SET(e.l).name, FIND_SET(e.r).name
      UNION(e.l, e.r)

TEST(V, E)
print "Finished!"
for v in V:
  print v.name, v.p.name, v.rank
