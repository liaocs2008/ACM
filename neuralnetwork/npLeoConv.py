import sys, os
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


def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            #download(filename)
            print ("MNIST DATA NOT FOUND")
            assert 1==0
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]





def test(layers, cost):
    B = 1  # batch size
    nlayers = len(layers)
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    acc = 0.
    num = 0
    for batch in iterate_minibatches(X_test, y_test, B, shuffle=False):
        x, y = batch
        x = x.reshape([B, 1, 28, 28])
        y = y.reshape([B])
        inputs = [x]
        for i in xrange(nlayers):
            inputs.append(layers[i].forward(inputs[-1]))  # inputs[i] is the input for i-th layer
        loss = cost.forward(inputs[-1], y)
        acc += np.sum(np.equal(np.argmax(cost.prob, axis=1), np.ravel(y)))
        num += B
        print(num, loss, float(acc)/num)

    acc = acc / num
    print "test acc=", acc




if __name__ == "__main__":
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


    # prepare numpy design
    batch_size = 1
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
    loss = SoftmaxCrossEntropyLoss()
    test(layers, loss)



