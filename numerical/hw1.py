#!/usr/bin/python

import numpy

def DFT_POLY(coefficent, x):
  N = len(coefficent)
  if N > 2:
    y = x * x

    even_c = [coefficent[i] for i in xrange(0, N, 2)]
    P_0 = DFT_POLY(even_c, y)

    odd_c = [coefficent[i] for i in xrange(1, N, 2)]
    P_1 = DFT_POLY(odd_c, y)

    return P_0 + x * P_1
  else:
    return coefficent[0] + coefficent[1] * x

def DFT_POLY2(coefficent, x):
  N = len(coefficent)
  if N > 2:
    y = x * x

    even_c = [coefficent[i] for i in xrange(0, N, 2)]
    P_0 = DFT_POLY(even_c, y)

    odd_c = [coefficent[i] for i in xrange(1, N, 2)]
    # save half of computation
    M = len(x) / 2
    odd_x = numpy.array([x[i] for i in xrange(0, M)], dtype='object')
    P_1 = DFT_POLY(odd_c, odd_x * odd_x)

    tmp = odd_x * P_1
    return P_0 + numpy.append(tmp, -tmp)
  else:
    return coefficent[0] + coefficent[1] * x

if __name__ == "__main__":
  
  coefficient = numpy.array([0, 1, 2, 3])
  w = numpy.array([1, 1j, -1, -1j])
  result = DFT_POLY2(coefficient, w)
  print result

  coefficient = [0, -1, 2, -3, 4, -5, 6, -7]
  t = numpy.sqrt(2) / 2
  w = numpy.array([1, t + t*1j, 1j, -t + t*1j, -1, -t - t*1j, -1j, t - t*1j], dtype='object')
  result = DFT_POLY2(coefficient, w)
  print result
