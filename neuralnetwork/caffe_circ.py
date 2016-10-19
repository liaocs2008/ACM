# coding: utf-8

caffe_root = '/home/leo/code/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import numpy as np

import sys, os

sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe.proto import caffe_pb2

from newcircfc.new_circfc import NewCircFC

from gradient_check_util import GradientChecker


import unittest, tempfile


from scipy.linalg import circulant


def init_w(shape):
    return np.random.normal(0, 1. / np.product(shape), shape)


class CircFC(object):
    def __init__(self, I, H, name=None, r=None, dr=None):
        # I : input size
        # H : hidden size
        self.k = max(I, H) # r is now padded
        #self.k = int( 2 ** np.ceil(np.log(self.k)/np.log(2)) )

        if r is not None and dr is not None:
            self.r = r
            self.dr = dr
        else:
            self.r = init_w( self.k )
            self.dr = np.zeros(self.k)
        assert r is not None
        assert dr is not None


        self.mapping = np.roll(np.arange(self.k)[::-1], 1) # for shifting x
        self.name = name
        self.H = H # desired output size

    def forward(self, x):
        # a = dot(x, R.T), shape=(b, H)
        if self.k > x.shape[1]: # then self.k = self.H
            # pad data
            # http://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
            # http://stackoverflow.com/questions/19349410/how-to-pad-with-zeros-a-tensor-along-some-axis-python
            assert self.k == self.H

            # start = time.time()
            # new_x = np.pad(x, [(0,0), (0, self.k-x.shape[1])], 'constant', constant_values=0)
            # end = time.time()
            # print "new_x", end - start
            #
            # start = time.time()
            # a = np.fft.ifft( np.fft.fft(new_x) * np.fft.fft(self.r) )[:,:self.H]
            # end = time.time()
            # print "a", end - start
            #
            # start = time.time()
            # b = np.fft.ifft( np.fft.fft(x, n=self.k) * np.fft.fft(self.r, n=self.k) , n = self.k )[:,:self.H]
            # end = time.time()
            # print "b", end - start
            #
            # assert np.all(a == b)

            #start = time.time()
            a = np.fft.ifft(np.fft.fft(x, n=self.k) * np.fft.fft(self.r, n=self.k), n=self.k)[:, :self.H]
            #end = time.time()
            #print "a", end - start

        else:
            #start = time.time()
            a = np.fft.ifft( np.fft.fft(x) * np.fft.fft(self.r) )[:, :self.H]
            #end = time.time()
            #print "a", end - start
        if __debug__:
            if self.k > x.shape[1]:
                new_x = np.pad(x, [(0,0), (0, self.k-x.shape[1])], 'constant', constant_values=0)
                check_a = np.dot(new_x, circulant(self.r).T)[:, :self.H]
            else:
                check_a = np.dot(x, circulant(self.r).T)[:, :self.H]
            assert np.linalg.norm( a - check_a ) < 1e-6
            print self.name, "forward", x.shape, "to", a.shape
        return np.real(a)

    def backward(self, x, d_a):
        if self.k > x.shape[1]:
            # pad data, don't forget to change form of x
            # this padding cannot be avoided since there are 0s "mapped" to different spaces
            assert self.k == self.H
            new_x = np.pad(x, [(0,0), (0, self.k-x.shape[1])], 'constant', constant_values=0)[:, self.mapping]
            self.dr = np.sum( np.fft.ifft(np.fft.fft(new_x)*np.fft.fft(d_a)), axis=0 )
            d_x = np.fft.ifft( np.fft.fft(d_a) * np.fft.fft(self.r[self.mapping]) )[:,:x.shape[1]]
        else:
            # start = time.time()
            # new_x = x[:, self.mapping]
            # end = time.time()
            # print "new_x", end-start
            #
            # start = time.time()
            # new_d_a = np.pad(d_a, [(0,0), (0, self.k-d_a.shape[1])], 'constant', constant_values=0)
            # end = time.time()
            # print "new_d_a", end-start
            #
            # start = time.time()
            # self.dr = np.sum( np.fft.ifft(np.fft.fft(new_x)*np.fft.fft(new_d_a)), axis=0 )
            # end = time.time()
            # print "self.dr", end - start
            #
            # start = time.time()
            # d_x = np.fft.ifft( np.fft.fft(new_d_a) * np.fft.fft(self.r[self.mapping]) )[:,:x.shape[1]]
            # end = time.time()
            # print "d_x", end - start
            #
            #
            # start = time.time()
            # tmp_dr = np.sum(np.fft.ifft(np.fft.fft(x[:,self.mapping], n=self.k) * np.fft.fft(d_a, n=self.k), n=self.k), axis=0)
            # end = time.time()
            #
            # start = time.time()
            # tmp_d_x = np.fft.ifft(np.fft.fft(d_a, n=self.k) * np.fft.fft(self.r[self.mapping], n=self.k), n=self.k)[:, :x.shape[1]]
            # end = time.time()
            #
            # assert np.all(tmp_dr == self.dr)
            # assert np.all(tmp_d_x == d_x)


            #start = time.time()
            self.dr = np.sum(np.fft.ifft(np.fft.fft(x[:,self.mapping], n=self.k) * np.fft.fft(d_a, n=self.k), n=self.k), axis=0)
            #end = time.time()
            #print "self.dr", end - start

            #start = time.time()
            d_x = np.fft.ifft(np.fft.fft(d_a, n=self.k) * np.fft.fft(self.r[self.mapping], n=self.k), n=self.k)[:, :x.shape[1]]
            #end = time.time()
            #print "d_x", end - start

        if __debug__:
            if self.k > x.shape[1]:
                check_d_x = np.dot(d_a, circulant(self.r))[:, :x.shape[1]]
            else:
                new_d_a = np.pad(d_a, [(0,0), (0, self.k-d_a.shape[1])], 'constant', constant_values=0)
                check_d_x = np.dot(new_d_a, circulant(self.r))[:, :x.shape[1]]
            diff_norm = np.linalg.norm( d_x - check_d_x )
            print self.name, "backward", d_a.shape, "to", d_x.shape, diff_norm
            assert diff_norm < 1e-9
        self.dr = np.real(self.dr)
        return np.real(d_x)

    def update(self, lr=0.01):
        self.r = self.r - lr * self.dr
        self.dr.fill(0.)



class NewCircFC(object):
    def __init__(self, I, H, r, dr, name=None):
        # I : input size
        # H : hidden size
        self.I = I
        self.H = H
        if self.I <= self.H:
            self.k = I
            self.c = [CircFC(self.k, self.k, name + '_tmp_%d' % i,
                             r=r[i*self.k:(i+1)*self.k],
                             dr=dr[i*self.k:(i+1)*self.k]) if (i + 1) * self.k <= self.H
                      else CircFC(self.k, self.H - i * self.k, name + '_tmp_%d' % i,
                             r=r[i*self.k:(i+1)*self.k],
                             dr=dr[i*self.k:(i+1)*self.k])
                      for i in xrange((self.H + self.I - 1) / self.I)]
        else:
            self.k = H
            self.c = [CircFC(self.k, self.k, name + '_tmp_%d' % i,
                             r=r[i*self.k:(i+1)*self.k],
                             dr=dr[i*self.k:(i+1)*self.k]) if (i + 1) * self.k <= self.I
                      else CircFC(self.I - i * self.k, self.k, name + '_tmp_%d' % i,
                                  r=r[i*self.k:(i+1)*self.k],
                                  dr=dr[i * self.k:(i + 1) * self.k])
                      for i in xrange((self.I + self.H - 1) / self.H)]
        self.name = name
        print "ATTENTION: You are using NewCircFC"

    def forward(self, x):
        assert self.I == x.shape[1]
        a = np.zeros((x.shape[0], self.H))
        if self.I <= self.H:
            for i in xrange((self.H + self.I - 1) / self.I): # ceil(5/3)=2, i={0,1} -- {0,3}
                end = (i + 1) * self.k
                if end > self.H:
                    end = self.H
                a[:, i*self.k:end] = self.c[i].forward(x)
        else:
            for i in xrange((self.I + self.H - 1) / self.H):
                end = (i + 1) * self.k
                if end > self.I:
                    end = self.I
                a += self.c[i].forward(x[:,i*self.k:end])
        if __debug__:
            print self.name, "forward", x.shape, "to", a.shape
        return a

    def backward(self, x, d_a):
        d_x = np.zeros(x.shape)
        if self.I <= self.H:
            for i in xrange((self.H + self.I - 1) / self.I): # ceil(5/3)=2, i={0,1} -- {0,3}
                end = (i + 1) * self.k
                if end > self.H:
                    end = self.H
                d_x += self.c[i].backward(x, d_a[:,i*self.k:end])
        else:
            for i in xrange((self.I + self.H - 1) / self.H):
                end = (i + 1) * self.k
                if end > self.I:
                    end = self.I
                d_x[:, i*self.k:end] = self.c[i].backward(x[:,i*self.k:end], d_a)
        if __debug__:
            print self.name, "backward", d_a.shape, "to", d_x.shape
        return d_x

    def update(self, lr=0.01):
        for i in xrange(len(self.c)):
            self.c[i].update(lr)


class C_FC(caffe.Layer):

    def SetUp(self, bottom, top):
        self.setup(bottom, top)

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1 or len(top) != 1:
            raise Exception("one to one")

        self.I = int(np.product(bottom[0].shape[1:]))
        self.H = int(self.param_str) #int(np.product(top[0].shape[1:]))

        if self.I <= self.H:
            self.data = np.zeros( self.I * ((self.H + self.I - 1) / self.I) )
        else:
            self.data = np.zeros( self.H * ((self.I + self.H - 1) / self.H) )

        self.diff = np.zeros_like(self.data)

        self.fc = NewCircFC(self.I, self.H, self.data, self.diff, name="circ")

        print "leo, self.I=", self.I, "self.H=", self.H


    def Reshape(self, bottom, top):
        self.reshape(bottom, top)

    def reshape(self, bottom, top):
        # dir(top[0]), 'channels', 'count', 'data', 'diff', 'height', 'num', 'reshape', 'shape', 'width'
        if len(bottom) != 1 or len(top) != 1:
            raise Exception("one to one")

        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        #print dir(top[0])
        top[0].reshape(bottom[0].num, self.H)

    def Forward(self, bottom, top):
        self.forward(bottom, top)

    def forward(self, bottom, top):
        # print "leo forward bottom", bottom[0].num, bottom[0].channels, bottom[0].width, bottom[0].height
        # print "leo forward top", top[0].num, top[0].channels, top[0].width, top[0].height
        top[0].data[...] = self.fc.forward(
                                bottom[0].data[...].reshape(bottom[0].num, bottom[0].channels * bottom[0].width * bottom[0].height)
                           ).reshape(*top[0].shape)


    def Backward(self, top, propagate_down, bottom):
        self.backward(top, propagate_down, bottom)

    def backward(self, top, propagate_down, bottom):
        # print "leo backward", propagate_down[0]
        # print "leo backward top", top[0].num, top[0].channels, top[0].width, top[0].height
        # print "leo backward bottom", bottom[0].num, bottom[0].channels, bottom[0].width, bottom[0].height
        if propagate_down[0]:
            bottom[0].diff[...] = self.fc.backward(
                                    bottom[0].data.reshape(bottom[0].num, bottom[0].channels * bottom[0].width * bottom[0].height),
                                    top[0].diff
                                  ).reshape(*bottom[0].shape)



def python_param_net_file():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write("""
        name: 'pythonnet'
        force_backward: true
        input: 'data' input_shape { dim: 3 dim: 2 dim: 16 dim: 16 }
        layer {
            type: 'Python'
            name: 'mul10'
            bottom: 'data'
            top: 'mul10'
            python_param {
                module: 'Conv'
                layer: 'C_FC'
                param_str: '100'
            }
        }
        layer {
            type: 'Python'
            name: 'mul2'
            bottom: 'mul10'
            top: 'mul2'
            python_param {
                module: 'Conv'
                layer: 'C_FC'
                param_str: '10'
            }
        }
        """)
        return f.name


@unittest.skipIf('Python' not in caffe.layer_type_list(),
    'Caffe built without Python layer support')
class TestLayerWithParam(unittest.TestCase):
    def setUp(self):
        net_file = python_param_net_file()
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    def test_forward(self):
        x = np.random.random([3,2,16,16])
        self.net.blobs['data'].data[...] = x
        self.net.forward()

    def test_backward(self):
        x = np.random.random([3,10])
        self.net.blobs['mul2'].diff[...] = x
        self.net.backward()

    def test_gradient(self):
        self.net.blobs['data'].data[...] = np.random.random([3,2,16,16])
        x = self.net.blobs['data']
        self.net.forward()
        self.net.blobs['mul2'].diff[...] = np.random.random([3, 10])
        y = self.net.blobs['mul2']

        self.net.backward()
        checker = GradientChecker(1e-5, 1e-5)
        checker.check_gradient_exhaustive(
            self.net.layers[1], [x], [y], check_bottom=[0])






if __name__ == '__main__':
    print "WARNING, test with support of chaning _caffe.cpp under caffe/python"
    unittest.main()

