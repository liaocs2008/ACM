#!/bin/python
def INSERT_SORT(a):
  for j in range(1,len(a)):
    key = a[j]
    i = j-1
    while i>=0 and a[i] > key:
      a[i+1] = a[i]
      i = i-1
    a[i+1] = key

if __name__ == "__main__":
  a = [3,4,5,6,1,2,0,8,9,7]
  INSERT_SORT(a)
  print a
