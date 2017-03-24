import numpy as np


def init_w(shape):
    return np.random.normal(0, 1. / np.product(shape), shape)


class BlockCircConv(object):
    def __init__(self, (iN, iC, iW, iD), (oN, oC, oW, oD), name=None):
        assert iN == oN
        self.w = np.asarray([iW, iW - oW + 1, oW])
        self.h = np.asarray([iD, iD - oD + 1, oD])
        self.c = np.asarray([iC, -1, oC])
        self.mc = np.max(self.c)

        self.f = init_w([self.mc, self.w[1] * self.h[1]])
	self.bias = init_w([self.mc])

        self.indx = np.arange(self.w[1] * self.h[1])[:,None] + np.arange(self.mc)*self.w[1] * self.h[1]
        self.indx = self.indx.ravel()

        start_idx = np.arange(self.w[1])[:, None] * self.h[0] + np.arange(self.h[1])
        offset_idx = np.arange(self.w[2])[:, None] * self.h[0] + np.arange(self.h[2])
        self.indx2 = np.arange(iN * self.c[0])[:, None] * (self.w[0] * self.h[0]) + \
                     (offset_idx.ravel()[::1][:,None] + start_idx.ravel()).ravel()
        self.name = name

    def blockcircmul(self, x, bc):
        #cols, rows, k = bc.shape
        k = self.mc
	cols = self.w[1] * self.h[1]
        rows = 1

        #print("blockcircmul", bc.shape, [1, cols, rows, k], self.c)
        bc = bc.reshape([1, cols, rows, k])
        a = np.zeros((x.shape[0], k * rows))
        new_x = np.pad(x, [(0, 0), (0, k * cols - x.shape[1])], 'constant', constant_values=0)
        new_x = new_x.reshape([x.shape[0], cols, 1, k])
        a = np.sum(
            np.fft.irfft(np.fft.rfft(new_x, n=k) * np.fft.rfft(bc, n=k), n=k),
            axis=1
        ).reshape(a.shape)
        return a

    def forward(self, x):
        batch, _, _, _ = x.shape
        target_shape = [batch * self.w[2] * self.h[2], self.c[0] * self.w[1] * self.h[1]]
        new_x = np.zeros([batch * self.w[2] * self.h[2], self.mc * self.w[1] * self.h[1]])
        new_x[:, :target_shape[1]] = x.ravel()[self.indx2.ravel()].reshape(batch, self.c[0], self.w[2] * self.h[2], self.w[1] * self.h[1]).transpose([0, 2, 1, 3]).reshape(target_shape)
        a = self.blockcircmul(new_x[:, self.indx], self.f)
        if __debug__:
            print self.name, "forward", x.shape, "to", a.shape
        a = a.reshape([x.shape[0], self.w[2] * self.h[2], self.mc])
        a = a.transpose([0, 2, 1]).reshape([x.shape[0], self.mc, self.w[2], self.h[2]])
        conved = a[:, :self.c[2], :, :]
        # dimshuffle('x', 0, 'x', 'x') 
        # http://deeplearning.net/software/theano/library/tensor/basic.html
        return conved + self.bias.reshape([1, self.mc, 1, 1]) 




class MaxPooling(object):
    def __init__(self, (iN, iC, iW, iD), (oN, oC, oW, oD), name=None):
        assert iN == oN
        self.w = np.asarray([iW, iW / oW, oW])
        self.h = np.asarray([iD, iD / oD, oD])
        self.c = np.asarray([iC, -1, oC])
        self.mc = np.max(self.c)

        self.indx = np.arange(self.w[1] * self.h[1])[:,None] + np.arange(self.mc)*self.w[1] * self.h[1]
        self.indx = self.indx.ravel()

        start_idx = np.arange(self.w[1])[:, None] * self.h[0] + np.arange(self.h[1])
        offset_idx = np.arange(self.w[2])[:, None] * self.h[0] * self.w[1] + np.arange(self.h[2]) * self.h[1]
        self.indx2 = np.arange(iN * self.c[0])[:, None] * (self.w[0] * self.h[0]) + \
                     (offset_idx.ravel()[:,None] + start_idx.ravel()).ravel()
        self.name = name
        print self.w, self.h, self.indx2.shape

    def forward(self, x):
        batch, _, _, _ = x.shape
        new_x = x.ravel()[self.indx2.ravel()].reshape(batch, self.c[0], self.w[2] * self.h[2], self.w[1] * self.h[1])
        if __debug__:
            print self.name, "forward", x.shape, "to", new_x.shape
        return np.amax(new_x, axis=-1).reshape(batch, self.c[0], self.w[2], self.h[2])



class BlockCircFC(object):
    def __init__(self, I, H, k, name=None):
        # I : input size
        # H : hidden size
        self.I = I
        self.H = H
        self.k = k
        self.rows = (self.H + self.k - 1) / self.k
        self.cols = (self.I + self.k - 1) / self.k
        self.w = init_w([1, self.cols, self.rows, self.k])
        self.mapping = np.roll(np.arange(self.k)[::-1], 1) # for shifting x
        self.name = name
        print "ATTENTION: You are using BlockCircFC"

    def forward(self, x):
        x = x.reshape([x.shape[0], -1])
        assert self.I == x.shape[1]
        a = np.zeros((x.shape[0], self.k*self.rows))
        new_x = np.pad(x, [(0,0), (0, self.k*self.cols-x.shape[1])], 'constant', constant_values=0)
        new_x = new_x.reshape([x.shape[0], self.cols, 1, self.k]) 
        if __debug__:
            print self.name, "forward", x.shape, "to", a.shape
        a = np.sum(
                np.fft.irfft(np.fft.rfft(new_x, n=self.k) * np.fft.rfft(self.w, n=self.k), n=self.k),
                axis=1
            ).reshape(a.shape)
        return a[:,:self.H]


class ReLU(object):
    def __init__(self, name=None):
        self.name = name

    def forward(self, a):
        b = np.copy(a)
        b[b <= 0] = 0
        if __debug__:
            print self.name, "forward", a.shape, "to", b.shape
        return b


class FC(object):
    def __init__(self, I, H, name=None):
        # I : input size
        # H : hidden size
        self.w = init_w([H, I])
        self.dw = np.zeros(self.w.shape)
        self.c = init_w([H, 1])
        self.dc = np.zeros(self.c.shape)

        self.name = name

    def forward(self, x):
        a = np.dot(x, self.w.T) + self.c.T # a = wx + c
        if __debug__:
            print self.name, "forward", x.shape, "to", a.shape
        return a


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name=None):
        self.name = name
        self.prob = None

    def forward(self, x, target): # example, x = [0.6,-0.4,0,2.1], target = 3
        e_x = np.exp(x - x.max(axis=1)[:, None])
        self.prob = e_x / e_x.sum(axis=1)[:, None]
        cost =  np.sum( - np.log(np.maximum(self.prob[np.arange(x.shape[0]), np.ravel(target)], 1e-6) ) )
        if __debug__:
            print "SoftmaxCrossEntropyLoss", cost
        return cost



import theano
import theano.tensor as T
import lasagne
from LeoConv import LeoConv2, LeoLayer4

def build_newcirc_cnn(input_var=None, input_shape=(500, 1, 28, 28)):
    network = lasagne.layers.InputLayer(shape=input_shape,
                                        input_var=input_var)

    network = LeoConv2(
          network, num_filters=32, filter_size=(5, 5),
          W=lasagne.init.GlorotUniform(),
          nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = LeoConv2(
        network, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = LeoLayer4(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            partition_size=256,
            nonlinearity=lasagne.nonlinearities.rectify, name="leolayer4_1")

    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax, name="leolayer4_2")

    return network




def test():
    x = np.random.random([16, 512]).astype(np.float32)
    w = np.random.random([2,256]).astype(np.float32)

    blockcircfc = BlockCircFC(512, 256, 256, 'blockcircfc')
    blockcircfc.w = w.reshape([1,2,1,256])
    res1 = blockcircfc.forward(x)
 
    
    input_var = T.matrix('inputs', dtype=theano.config.floatX)
    network = lasagne.layers.InputLayer(shape=(16, 512),
                                        input_var=input_var)
    network = LeoLayer4(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            partition_size=256,
            nonlinearity=lasagne.nonlinearities.rectify, name="leolayer4")
    out = lasagne.layers.get_output(network, deterministic=True)
    lasagne.layers.set_all_param_values(network, [w.reshape([1,2,256])])
    fn = theano.function([input_var], [out])
    res2 = fn(x)
    print(np.allclose(res1, res2), res1, res2)


def test2():
    x = np.random.random([1, 256]).astype(np.float32)
    w = np.random.random([10, 256]).astype(np.float32)
    b = np.random.random([10]).astype(np.float32)

    blockcircfc = FC(256, 10, 'fc')
    blockcircfc.w = w.reshape([10,256])
    blockcircfc.c = b
    res1 = blockcircfc.forward(x)
 
    
    input_var = T.matrix('inputs', dtype=theano.config.floatX)
    network = lasagne.layers.InputLayer(shape=(1, 256),
                                        input_var=input_var)
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.rectify)
    out = lasagne.layers.get_output(network, deterministic=True)
    lasagne.layers.set_all_param_values(network, [w.T, b])
    fn = theano.function([input_var], [out])
    res2 = fn(x)
    print(np.allclose(res1, res2), res1, res2)






if __name__ == "__main__":
    #test()
    #test2()

    #"""
    batch_size = 1
    x = np.random.random([batch_size, 1, 28, 28]).astype(np.float32)

    # arr_0 (32,)
    # arr_1 (25, 32)
    # arr_2 (32,)
    # arr_3 (25, 32)
    # arr_4 (1, 2, 256)
    # arr_5 (256, 10)
    # arr_6 (10,)
    f = np.load('leoconvmodel.npz')
    for k in f:
        print(k, f[k].shape)


    # prepare lasagne design 
    input_var = T.tensor4('inputs')
    network = build_newcirc_cnn(input_var, input_shape=(batch_size,1,28,28))
    prediction = lasagne.layers.get_output(network, deterministic=True)
    T_fn = theano.function([input_var], prediction)

    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    #param_values = [f['arr_0'], f['arr_1'], f['arr_2'], f['arr_3'], f['arr_4']]
    lasagne.layers.set_all_param_values(network, param_values)
    T_y = T_fn(x)



    # prepare numpy design
    inputs = [x]
    layers = [BlockCircConv([batch_size, 1, 28, 28], [batch_size, 32, 24, 24], 'conv0'), 
	      ReLU('relu0'), 
	      MaxPooling([batch_size, 32, 24, 24], [batch_size, 32, 12, 12], 'pool1'),
	      BlockCircConv([batch_size, 32, 12, 12], [batch_size, 32, 8, 8], 'conv1'), 
              ReLU('relu1'),
	      MaxPooling([batch_size, 32, 8, 8], [batch_size, 32, 4, 4], 'pool2'),
              BlockCircFC(512, 256, 256, 'fc1'),
              ReLU('relu2'),
	      FC(256, 10, 'fc1'),
              ReLU('relu3')]
    layers[0].bias = f['arr_0']
    layers[0].f = f['arr_1'] 
    layers[3].bias = f['arr_2']
    layers[3].f = f['arr_3'] 
    layers[6].w = f['arr_4'].reshape(layers[6].w.shape)
    layers[8].w = f['arr_5'].T
    layers[8].c = f['arr_6']
    nlayers = len(layers)
    for i in xrange(nlayers):
        inputs.append(layers[i].forward(inputs[-1]))  # inputs[i] is the input for i-th layer
    loss = SoftmaxCrossEntropyLoss()
    loss.forward(inputs[-1], np.array([0,1,0,0,0,0,0,0,0,0]))
    y = loss.prob


    y = y.reshape(T_y.shape)
    print( np.allclose(y, T_y), np.abs(y - T_y).max() )
    #"""


