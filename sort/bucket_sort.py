#!/bin/python
t = __import__('insertion-sort')
def BUCKET_SORT(a, n):
  b = []
  for i in range(n):
    b.append([])
  for i in range(n):
    b[int(n*a[i])].append(a[i])
  s = []
  for i in range(n):
    t.INSERT_SORT(b[i])
    s = s + b[i]
  return s

a = [.78, .17, .39, .26, .72, .94, .21, .12, .23, .68]
s = BUCKET_SORT(a, len(a))
print s
