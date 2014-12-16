#!/bin/python
def MAX_HEAPIFY(a, i, n):
  largest = i
  left = 2*i
  right = 2*i+1
  if left < n and a[largest] < a[left]:
    largest = left
  if right < n and a[largest] < a[right]:
    largest = right
  if largest != i:
    tmp = a[i]
    a[i] = a[largest]
    a[largest] = tmp
    MAX_HEAPIFY(a, largest, n)

def BUILD_MAX_HEAP(a, n):
  for i in range(n/2, -1, -1):
    MAX_HEAPIFY(a, i, n)

def HEAP_SORT(a, n):
  BUILD_MAX_HEAP(a,n)
  for i in range(n-1, -1, -1):
    maximum = a[0]
    a[0] = a[i]
    a[i] = maximum
    MAX_HEAPIFY(a, 0, i)

a = [1,3,9,8,7,6,4,5,3,0]
HEAP_SORT(a, len(a))
print a


def HEAP_EXTRACT_MAX(a, n):
  maximum = a[0]
  a[0] = a[n-1]
  n = n - 1
  a.pop() # remove last from list
  MAX_HEAPIFY(a, 0, n)
  return maximum

def HEAP_INCREASE_KEY(a, i, key):
  a[i] = key
  parent = i/2
  while i>=0 and a[parent] < a[i]:
    tmp = a[i]
    a[i] = a[parent]
    a[parent] = tmp
    i = parent
    parent = i/2

def MAX_HEAP_INSERT(a, key, n):
  n = n+1
  a.append(-float('Inf')) # a[n-1] = -float('Inf')
  HEAP_INCREASE_KEY(a, n-1, key)

a = []
print a
MAX_HEAP_INSERT(a, 1, len(a))
print a
HEAP_EXTRACT_MAX(a, len(a))
print a

