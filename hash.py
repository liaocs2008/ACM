#!/bin/python
def DIRECT_ADDRESS_SEARCH(T, k):
  return T[k]

def DIRECT_ADDRESS_INSERT(T, x):
  T[x['key']] = x

def DIRECT_ADDRESS_DELETE(T, x):
  T[x['key']] = None

m = 10
T = [[] for i in range(m)]

x1 = {'key': 2, 'data': 'abc'}
x2 = {'key': 3, 'data': 2}
DIRECT_ADDRESS_INSERT(T, x1)
DIRECT_ADDRESS_INSERT(T, x2)
print T
print DIRECT_ADDRESS_SEARCH(T, 2)
DIRECT_ADDRESS_DELETE(T, x1)
print T

def HASH(k):
  return k % m

def CHAINED_HASH_INSERT(T, x):
  T[HASH(x['key'])].insert(0, x)

def CHAINED_HASH_SEARCH(T, k):
  for x in T[HASH(k)]:
    if x['key'] == k:
      return x
  return None

def CHAINED_HASH_DELETE(T, x):
  T[HASH(x['key'])].remove(x)

m = 10
T = [[] for i in range(m)]
x1 = {'key': 2, 'data': 'abc'}
x2 = {'key': 3, 'data': 2}
x3 = {'key': 2, 'data': 3}
CHAINED_HASH_INSERT(T, x1)
CHAINED_HASH_INSERT(T, x2)
CHAINED_HASH_INSERT(T, x3)
print T
print CHAINED_HASH_SEARCH(T, 2)
print CHAINED_HASH_SEARCH(T, 0)
CHAINED_HASH_DELETE(T, x3)
print T
