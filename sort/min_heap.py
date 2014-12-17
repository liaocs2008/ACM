#!/bin/python
def MIN_HEAPIFY(a, i, n):
  minimum = i
  left = 2*i
  right = 2*i+1
  if left < n and a[minimum] > a[left]:
    minimum = left
  if right < n and a[minimum] > a[right]:
    minimum = right
  if minimum != i:
    tmp = a[i]
    a[i] = a[minimum]
    a[minimum] = tmp
    MIN_HEAPIFY(a, minimum, n)

def BUILD_MIN_HEAP(a, n):
  for i in range(n/2, -1, -1):
    MIN_HEAPIFY(a, i, n)

def HEAP_SORT(a, n):
  BUILD_MIN_HEAP(a,n)
  for i in range(n-1, -1, -1):
    minimum = a[0]
    a[0] = a[i]
    a[i] = minimum
    MIN_HEAPIFY(a, 0, i)


def HEAP_EXTRACT_MIN(a, n):
  minimum = a[0]
  a[0] = a[n-1]
  n = n - 1
  a.pop() # remove the tail
  MIN_HEAPIFY(a, 0, n)
  return minimum

def HEAP_DECREASE_KEY(a, i, key):
  a[i] = key
  parent = i/2
  while i>=0 and a[parent] > a[i]:
    tmp = a[i]
    a[i] = a[parent]
    a[parent] = tmp
    i = parent
    parent = i/2

def MIN_HEAP_INSERT(a, key, n):
  n = n+1
  a.append(key) # a[n-1] = -float('Inf')
  HEAP_DECREASE_KEY(a, n-1, key)

if __name__ == "__main__":
  a = [1,2,9,8,7,6,4,5,3,0]
  HEAP_SORT(a, len(a))
  print a
  a = []
  print a
  MIN_HEAP_INSERT(a, 1, len(a))
  MIN_HEAP_INSERT(a, 3, len(a))
  MIN_HEAP_INSERT(a, 2, len(a))
  MIN_HEAP_INSERT(a, 0, len(a))
  MIN_HEAP_INSERT(a, 4, len(a))
  MIN_HEAP_INSERT(a, 5, len(a))
  print a
  HEAP_EXTRACT_MIN(a, len(a))
  print a
