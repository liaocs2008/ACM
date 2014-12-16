#!/bin/python
def FIND_MAX_CROSSING_SUBARRAY(a, low, mid, high):
  left_sum = -float('Inf')
  max_left = -1
  s = 0
  for i in range(mid, low-1, -1):
    s = s + a[i]
    if s > left_sum:
      left_sum = s
      max_left = i
  right_sum = -float('Inf')
  max_right = -1
  s = 0
  for j in range(mid+1, high+1):
    s = s + a[j]
    if s > right_sum:
      right_sum = s
      max_right = j
  return (max_left, max_right, left_sum + right_sum)

def FIND_MAXIMUM_SUBARRAY(a, low, high):
  if low == high:
    return (low, high, a[low])
  else:
    mid = (low+high)/2
    (left_low, left_high, left_sum) = FIND_MAXIMUM_SUBARRAY(a, low, mid)
    (right_low, right_high, right_sum) = FIND_MAXIMUM_SUBARRAY(a, mid+1, high)
    (cross_low, cross_high, cross_sum) = FIND_MAX_CROSSING_SUBARRAY(a, low, mid, high)
    if left_sum >= right_sum and left_sum >= cross_sum:
      return (left_low, left_high, left_sum)
    elif right_sum >= left_sum and right_sum >= cross_sum:
      return (right_low, right_high, right_sum)
    else:
      return (cross_low, cross_high, cross_sum)

a = [2,3,-1,-4,5,-7,19,0,-8,9]
(low,high,s) = FIND_MAXIMUM_SUBARRAY(a, 0, len(a)-1)
print a
for i in range(low, high+1):
  print a[i]
print low, high, s
