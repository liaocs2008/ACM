#!/bin/python
V = [60,100,120]
W = [10, 20, 30]

def KNAPSACK_0_1(V, W, s):
  tmp = [0]
  for i in range(len(W)):
    if W[i] <= s and W[i] > 0:
      tmp_w = [w for w in W]
      tmp_w[i] = 0
      tmp_v = [v for v in V]
      tmp_v[i] = 0
      tmp.append(V[i] + KNAPSACK_0_1(tmp_v, tmp_w, s-W[i]))
  return max(tmp)

def FRACTIONAL_KNAPSACK(V, W, s):
  if s <=0 :
    return 0
  count = 0
  tmp = [0 for v in V]
  for i in range(len(V)):
    if W[i] > 0:
      tmp[i] = V[i] / W[i]
      count = count + 1
  if count <= 0:
    return 0
  i = tmp.index(max(tmp))
  if W[i] < s:
    take_away = W[i]
    value = take_away * tmp[i]
  else:
    take_away = s
    value = V[i]
  tmp_v = [v for v in V]
  tmp_v[i] = tmp_v[i] - value
  tmp_w = [w for w in W]
  tmp_w[i] = tmp_w[i] - take_away
  return value + FRACTIONAL_KNAPSACK(tmp_v, tmp_w, s-take_away)

T = [10,20,30,40,50,60,70,80,90,100]
for t in T:
  print KNAPSACK_0_1(V, W, t)
  print FRACTIONAL_KNAPSACK(V, W, t)
