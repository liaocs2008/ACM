import numpy as np

class BaseConv(object):
    """
    This is only for two matrix doing convolution
    """

    def __init__(self, I, name=None):
        # I : input size
        self.I = I
        self.w = init_w([self.I, self.I])
        self.dw = np.zeros(self.w.shape)

        self.name = name

    def forward(self, x):
        self.dw = 0  # this is to clear, corresponding to the accumulation in backward
        w = self.w.ravel()[::-1].reshape([self.I, self.I])
        s = [self.I + x.shape[0] - 1, self.I + x.shape[1] - 1]
        r = np.fft.irfft2(np.fft.rfft2(w, s) * np.fft.rfft2(x, s), s)[(self.I - 1):(x.shape[0]),
            (self.I - 1):(x.shape[1])]
        if __debug__:
            fr = np.zeros([x.shape[0] - self.w.shape[0] + 1, x.shape[1] - self.w.shape[1] + 1])
            for i in xrange(r.shape[0]):
                for j in xrange(r.shape[1]):
                    fr[i, j] = np.sum(self.w * x[i:i + self.w.shape[0], j:j + self.w.shape[1]])
            assert np.allclose(fr, r)
            print self.name, "forward", x.shape, "to", r.shape
        return r

    def backward(self, x, d_a):
        da = d_a.ravel()[::-1].reshape(d_a.shape)
        s = [da.shape[0] + x.shape[0] - 1, da.shape[1] + x.shape[1] - 1]
        dw = np.fft.irfft2(np.fft.rfft2(da, s) * np.fft.rfft2(x, s), s)
        dw = dw[(d_a.shape[0] - 1):(x.shape[0]), (d_a.shape[1] - 1):(x.shape[1])]
        self.dw += dw  # !!!!!! notice that this is for multi-channel

        s = [d_a.shape[0] + self.I - 1, d_a.shape[1] + self.I - 1]
        d_x = np.fft.irfft2(np.fft.rfft2(d_a, s) * np.fft.rfft2(self.w, s), s)

        if __debug__:
            fdw = np.zeros(self.dw.shape)
            for a in xrange(self.w.shape[0]):
                for b in xrange(self.w.shape[1]):
                    fdw[a, b] = np.sum(d_a * x[a:a + d_a.shape[0], b:b + d_a.shape[1]])
            assert np.allclose(fdw, dw)

            fdx = np.zeros(x.shape)  # this is for next backpropagate layer
            assert d_a.shape[0] + self.w.shape[0] - 1 == d_x.shape[0]
            assert d_a.shape[1] + self.w.shape[1] - 1 == d_x.shape[1]
            for i in xrange(d_a.shape[0]):
                for j in xrange(d_a.shape[1]):
                    for a in xrange(self.w.shape[0]):
                        for b in xrange(self.w.shape[1]):
                            fdx[i + a, j + b] += d_a[i, j] * self.w[a, b]
            assert np.allclose(fdx, d_x)
            print self.name, "backward", d_a.shape, "to", d_x.shape
        return d_x

    def update(self, lr=0.01):
        self.w = self.w - lr * self.dw
        self.dw.fill(0.)


class Conv(object):
    def __init__(self, (iN, iC, iW, iD), (oN, oC, oW, oD), name=None):
        # (batch_size, channels, width, height)
        # assume stride is 1
        assert iN == oN
        filter_size = iW - oW + 1
        self.c = iC
        self.f = oC
        self.b = [[BaseConv(filter_size, (name + '_tmp_i%d_o%d' % (i, j))) for i in xrange(self.c)] for j in
                  xrange(self.f)]
        self.w = oW
        self.name = name

    def forward(self, x):
        a = np.zeros([x.shape[0], self.f, self.w, self.w])
        for i in xrange(a.shape[0]):
            for f in xrange(self.f):
                for c in xrange(self.c):
                    a[i, f, :, :] += self.b[f][c].forward(x[i, c, :, :])
        if __debug__:
            print self.name, "forward", x.shape, "to", a.shape
        return a

    def backward(self, x, d_a):
        d_x = np.zeros(x.shape)
        for i in xrange(d_a.shape[0]):
            for f in xrange(self.f):
                for c in xrange(self.c):
                    d_x[i, c, :, :] += self.b[f][c].backward(x[i, c, :, :], d_a[i, f, :, :])
        if __debug__:
            print self.name, "backward", d_a.shape, "to", d_x.shape
        return d_x

    def update(self, lr=0.01):
        for c in self.c:
            c.update(lr)


def test1():
    w1 = 2
    h1 = 2
    c = 3
    f = np.random.random([c, w1 * h1])

    a = np.arange(0, c).reshape([1, c])
    b = np.arange(0, -c, -1).reshape([c, 1])
    # print a+b
    ind = a + b
    # print f
    # print "==="
    cf = f[ind, :].reshape(c, c * w1 * h1).T
    # print cf

    indx = np.zeros(c * w1 * h1, dtype=np.int)
    for i in xrange(w1 * h1):
        indx[i * c:(i + 1) * c] = i + np.arange(0, c * w1 * h1, w1 * h1)
    # print indx
    print cf[indx, :]


def im2col_sliding_broadcasting(A, BSZ, stepsize=1):
    # Parameters
    M, N = A.shape
    col_extent = N - BSZ[1] + 1
    row_extent = M - BSZ[0] + 1
    # Get Starting block indices
    start_idx = np.arange(BSZ[0])[:, None] * N + np.arange(BSZ[1])
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    # Get all actual indices & index into input array for final output
    return np.take(A, start_idx.ravel()[:, None] + offset_idx.ravel()[::stepsize])


def leo_im2col(x, (w1, h1)):
    # don't support stride and padding now
    batch, c0, w0, h0 = x.shape
    w2 = w0 - w1 + 1
    h2 = h0 - h1 + 1

    start_idx = np.arange(w1)[:, None] * h0 + np.arange(h1)
    offset_idx = np.arange(w2)[:, None] * h0 + np.arange(h2)
    indx = start_idx.ravel()[:, None] + offset_idx.ravel()[::1]

    """
    a = np.zeros([c0 * w1 * h1, batch * w2 * h2])
    for b in xrange(batch):
        for c in xrange(c0):
            #a[c * w1 * h1:(c + 1) * w1 * h1, b * w2 * h2:(b + 1) * w2 * h2] = np.take(x[b, c, :,:], indx)
            a[c * w1 * h1:(c + 1) * w1 * h1, b * w2 * h2:(b + 1) * w2 * h2] = x[b, c, :, :].ravel()[indx.ravel()].reshape(indx.shape)
    return a.T
    """

    a2 = np.zeros([batch*w2*h2, c0*w1*h1])
    indx2 = np.arange(batch * c0)[:, None] * (w0 * h0) + indx.T.ravel()
    a2 = x.flatten()[indx2.ravel()].reshape(batch, c0, w2*h2, w1*h1).transpose([0,2,1,3]).reshape(a2.shape)

    return a2


def test2():
    x = np.random.random([1, 3, 7, 7])
    ind = leo_im2col(x, [3, 3])
    print ind.shape, x.shape


def test3():
    x = np.random.random([1, 3, 7, 7])
    ind = leo_im2col(x, [3, 3])
    print ind.shape, x.shape
    x2 = col2im_indices(ind, x.shape, 3, 3)


def init_w(shape):
    return np.random.normal(0, 1. / np.product(shape), shape)


def blockcircmul(x, bc):
    cols, rows, k = bc.shape
    bc = bc.reshape([1, cols, rows, k])
    a = np.zeros((x.shape[0], k * rows))
    new_x = np.pad(x, [(0, 0), (0, k * cols - x.shape[1])], 'constant', constant_values=0)
    new_x = new_x.reshape([x.shape[0], cols, 1, k])
    a = np.sum(
        np.fft.irfft(np.fft.rfft(new_x, n=k) * np.fft.rfft(bc, n=k), n=k),
        axis=1
    ).reshape(a.shape)
    return a


class BlockCircConv(object):
    def __init__(self, (iN, iC, iW, iD), (oN, oC, oW, oD), name=None):
        assert iN == oN
        self.filter_size = iW - oW + 1
        self.oC = oC
        self.oW = oW
        self.oD = oD
        self.iC = iC
        self.c = np.max([iC, oC])
        self.f = init_w([self.c, self.filter_size * self.filter_size])
        self.indx = np.zeros(self.c * self.filter_size * self.filter_size, dtype=np.int)
        for i in xrange(self.filter_size * self.filter_size):
            self.indx[i * self.c:(i + 1) * self.c] = i + np.arange(0, self.c * self.filter_size * self.filter_size,
                                                                   self.filter_size * self.filter_size)
        a = np.arange(0, self.c).reshape([1, self.c])
        b = np.arange(0, -self.c, -1).reshape([self.c, 1])
        self.indx2 = a + b
        self.name = name

    def forward(self, x):
        self.cf = self.f[self.indx2, :].reshape(self.c, self.c * self.filter_size * self.filter_size).T
        self.cf = self.cf[self.indx, :]
        new_x = leo_im2col(x, (self.filter_size, self.filter_size))
        assert new_x.shape == (x.shape[0] * self.oW * self.oD, self.filter_size * self.filter_size * self.iC)
        new_x = np.pad(new_x, [(0, 0), (0, (self.c - self.iC) * self.filter_size * self.filter_size)], 'constant',
                       constant_values=(0, 0))
        # a = np.dot(new_x[:,self.indx], self.cf)

        #print "cf", self.cf
        self.f = self.f.T.transpose([1, 0, 2])
        #print "f", self.f
        self.lf = self.f.reshape([self.filter_size * self.filter_size, 1, self.c])[:, :,
                  np.roll(np.arange(self.c)[::-1], 1)]
        a = blockcircmul(new_x[:, self.indx], self.lf)
        assert np.allclose(np.dot(new_x[:, self.indx], self.cf), a)

        if __debug__:
            print self.name, "forward", x.shape, "to", a.shape
        a = a.reshape([x.shape[0], self.oW * self.oD, self.c])
        a = a.transpose([0, 2, 1]).reshape([x.shape[0], self.c, self.oW, self.oD])
        return a[:, :self.oC, :, :]

    def backward(self, x, d_a):
        new_d_a = np.zeros([x.shape[0], self.c, self.oW, self.oD])
        new_d_a[:, :self.oC, :, :] = d_a
        new_d_a = new_d_a.reshape([x.shape[0], self.c, self.oW * self.oD])
        new_d_a = new_d_a.transpose([0, 2, 1])
        new_d_a = new_d_a.reshape(x.shape[0] * self.oW * self.oD, self.c)

        new_x = leo_im2col(x, (self.filter_size, self.filter_size))
        assert new_x.shape == (x.shape[0] * self.oW * self.oD, self.filter_size * self.filter_size * self.iC)
        new_x = np.pad(new_x, [(0, 0), (0, (self.c - self.iC) * self.filter_size * self.filter_size)], 'constant',
                       constant_values=(0, 0))
        self.df = np.dot(new_x.T, new_d_a)

        d_x = np.dot(new_d_a, self.cf.T)
        return d_x[:]

    def update(self, lr=0.01):
        pass


def test4():
    ic = 2
    oc = 3
    batch = 2
    # assert ic == oc
    x = np.random.random([batch, ic, 4, 4])
    conv = Conv(x.shape, (batch, oc, 2, 2), 'conv0')

    c = np.max([ic, oc])
    f = np.random.random([c, 3, 3])

    a = np.arange(0, c).reshape([1, c])
    b = np.arange(0, -c, -1).reshape([c, 1])
    cf = f[a + b, :, :]
    for i in xrange(oc):
        for j in xrange(ic):
            conv.b[i][j].w = cf[i, j, :, :]
    res0 = conv.forward(x)
    print res0.shape

    blockcircconv = BlockCircConv(x.shape, (batch, oc, 2, 2), 'conv1')
    blockcircconv.f = f
    res1 = blockcircconv.forward(x)
    print res1.shape

    print np.allclose(res0, res1)

    # d_a = np.random.random(res0.shape)
    # res2 = conv.backward(x, d_a)
    # print res2.shape
    # res3 = blockcircconv.backward(x, d_a)
    # print res3.shape
    # print np.allclose(res2, res3)



# lasagne use theano's conv2d
# theano use _convolve2d from scipy
# https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/conv.py, line 805
from scipy.signal.sigtools import _convolve2d
class BaseConv2(object):
    """
    This is only for two matrix doing convolution
    """

    def __init__(self, I, name=None):
        # I : input size
        self.I = I
        self.w = init_w([self.I, self.I])
        self.dw = np.zeros(self.w.shape)

        self.name = name
        print "warning! you are using BaseConv2"

    def forward(self, x):
        #r = _convolve2d(x, self.w, 1, val, bval, 0)
        r = _convolve2d(x, self.w.flatten()[::-1].reshape(self.w.shape), 1, 0, 0, 0) # Leo has no idea on meaning of these values
        if __debug__:
            fr = np.zeros([x.shape[0] - self.w.shape[0] + 1, x.shape[1] - self.w.shape[1] + 1])
            for i in xrange(r.shape[0]):
                for j in xrange(r.shape[1]):
                    fr[i, j] = np.sum(self.w * x[i:i + self.w.shape[0], j:j + self.w.shape[1]])
            assert np.allclose(fr, r)
            print self.name, "forward", x.shape, "to", r.shape
        return r

    def backward(self, x, d_a):
        assert False

    def update(self, lr=0.01):
        self.w = self.w - lr * self.dw
        self.dw.fill(0.)


class Conv2(object):
    def __init__(self, (iN, iC, iW, iD), (oN, oC, oW, oD), name=None):
        # (batch_size, channels, width, height)
        # assume stride is 1
        assert iN == oN
        filter_size = iW - oW + 1
        self.c = iC
        self.f = oC
        self.b = [[BaseConv2(filter_size, (name + '_tmp_i%d_o%d' % (i, j))) for i in xrange(self.c)] for j in
                  xrange(self.f)]
        self.w = oW
        self.name = name

    def forward(self, x):
        a = np.zeros([x.shape[0], self.f, self.w, self.w])
        for i in xrange(a.shape[0]):
            for f in xrange(self.f):
                for c in xrange(self.c):
                    a[i, f, :, :] += self.b[f][c].forward(x[i, c, :, :])
        if __debug__:
            print self.name, "forward", x.shape, "to", a.shape
        return a

    def backward(self, x, d_a):
        assert False

    def update(self, lr=0.01):
        for c in self.c:
            c.update(lr)



def test5():
    # verify that we could achieve the same result as scipy convolve2d which is the base of theano implementation
    ic = 2
    oc = 3
    batch = 2
    # assert ic == oc
    x = np.random.random([batch, ic, 4, 4]).astype(np.float32)
    conv = Conv2(x.shape, (batch, oc, 2, 2), 'conv0')

    c = np.max([ic, oc])
    f = np.random.random([c, 3, 3]).astype(np.float32)

    a = np.arange(0, c).reshape([1, c])
    b = np.arange(0, -c, -1).reshape([c, 1])
    cf = f[a + b, :, :]
    for i in xrange(oc):
        for j in xrange(ic):
            conv.b[i][j].w = cf[i, j, :, :]
    res0 = conv.forward(x)
    print res0.shape

    blockcircconv = BlockCircConv(x.shape, (batch, oc, 2, 2), 'conv1')
    blockcircconv.f = f
    res1 = blockcircconv.forward(x)
    print res1.shape
    print np.allclose(res0, res1)


    import lasagne
    import os
    os.environ["THEANO_FLAGS"] = "device=cpu"
    import theano
    import theano.tensor as T
    theano.config.optimizer='None'
    input_var = T.tensor4('inputs', dtype=theano.config.floatX)
    network = lasagne.layers.InputLayer(shape=(None, ic, 4, 4),
                                        input_var=input_var)
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=oc, filter_size=(3, 3),
        nonlinearity=None, pad='valid',
        W=np.fliplr(f.reshape(c, 3*3)).reshape(c, 3, 3)[a+b, :, :][:oc, :ic, :, :])
    out = lasagne.layers.get_output(network, deterministic=True)
    #lasagne.layers.set_all_param_values(network, cf)
    fn = theano.function([input_var], [out])
    res2 = fn(x)
    res2 = np.asarray(res2, dtype=np.float32)

    print "comp Conv2 with lasagne", np.allclose(res0, res2)
    print "comp BlockCircConv with lasagne", np.allclose(res1, res2)





class NeatBlockCircConv(object):
    def __init__(self, (iN, iC, iW, iD), (oN, oC, oW, oD), name=None):
        assert iN == oN
        self.w = np.asarray([iW, iW - oW + 1, oW])
        self.h = np.asarray([iD, iD - oD + 1, oD])
        self.c = np.asarray([iC, -1, oC])
        self.mc = np.max(self.c)
        self.f = init_w([self.mc, self.w[1] * self.h[1]])

        self.indx = np.arange(self.w[1] * self.h[1])[:,None] + np.arange(self.mc)*self.w[1] * self.h[1]
        self.indx = self.indx.ravel()

        start_idx = np.arange(self.w[1])[:, None] * self.h[0] + np.arange(self.h[1])
        offset_idx = np.arange(self.w[2])[:, None] * self.h[0] + np.arange(self.h[2])
        self.indx2 = np.arange(iN * self.c[0])[:, None] * (self.w[0] * self.h[0]) + \
                     (offset_idx.ravel()[::1][:,None] + start_idx.ravel()).ravel()
        self.name = name

    def blockcircmul(self, x, bc):
        cols, rows, k = bc.shape
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
        return a[:, :self.c[2], :, :]


def test6():
    # verify that we could achieve the same result as scipy convolve2d which is the base of theano implementation
    ic = 2
    oc = 3
    batch = 2
    # assert ic == oc
    x = np.random.random([batch, ic, 4, 4]).astype(np.float32)

    c = np.max([ic, oc])
    f = np.random.random([c, 3, 3]).astype(np.float32)


    blockcircconv = NeatBlockCircConv(x.shape, (batch, oc, 2, 2), 'conv1')
    ff = np.fliplr(f.reshape(c, 3 * 3)).reshape([c, 3, 3]).transpose([1, 2, 0])
    ff = ff.reshape([3 * 3, 1, c])[:, :, np.roll(np.arange(c)[::-1], 1)]
    blockcircconv.f = ff

    res1 = blockcircconv.forward(x)
    print res1.shape

    import lasagne
    import os
    os.environ["THEANO_FLAGS"] = "device=cpu"
    import theano
    import theano.tensor as T
    theano.config.optimizer='None'
    input_var = T.tensor4('inputs', dtype=theano.config.floatX)
    network = lasagne.layers.InputLayer(shape=(None, ic, 4, 4),
                                        input_var=input_var)
    a = np.arange(0, c).reshape([1, c])
    b = np.arange(0, -c, -1).reshape([c, 1])
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=oc, filter_size=(3, 3),
        nonlinearity=None, pad='valid',
        W=f[a+b, :, :][:oc, :ic, :, :])
    out = lasagne.layers.get_output(network, deterministic=True)
    #lasagne.layers.set_all_param_values(network, cf)
    fn = theano.function([input_var], [out])
    res2 = fn(x)
    res2 = np.asarray(res2, dtype=np.float32)

    print "comp BlockCircConv with lasagne", np.allclose(res1, res2)



class MaxPooling(object):
    def __init__(self, (iN, iC, iW, iD), (oN, oC, oW, oD), name=None):
        assert iN == oN
        self.w = np.asarray([iW, iW - oW , oW])
        self.h = np.asarray([iD, iD - oD , oD])
        self.c = np.asarray([iC, -1, oC])
        self.mc = np.max(self.c)

        self.indx = np.arange(self.w[1] * self.h[1])[:,None] + np.arange(self.mc)*self.w[1] * self.h[1]
        self.indx = self.indx.ravel()

        start_idx = np.arange(self.w[1])[:, None] * self.h[0] + np.arange(self.h[1])
        offset_idx = np.arange(self.w[2])[:, None] * self.h[0] + np.arange(self.h[2])
        self.indx2 = np.arange(iN * self.c[0])[:, None] * (self.w[0] * self.h[0]) + \
                     (offset_idx.ravel()[::(self.w[1]-1)][:,None] + start_idx.ravel()).ravel()
        self.name = name
        print self.w, self.h, self.indx2.shape

    def forward(self, x):
        batch, _, _, _ = x.shape
        new_x = np.zeros([batch * self.w[2] * self.h[2], self.c[0] * self.w[1] * self.h[1]])
        new_x = x.ravel()[self.indx2.ravel()].reshape(batch, self.c[0], self.w[2] * self.h[2], self.w[1] * self.h[1])
        if __debug__:
            print self.name, "forward", x.shape, "to", new_x.shape
        return np.amax(new_x, axis=3).reshape(batch, self.c[0], self.w[2], self.h[2]) 


def test7():
    c = ic = 2
    batch = 2
    x = np.random.random([batch, ic, 4, 4]).astype(np.float32)

    mp = MaxPooling(x.shape, (batch, ic, 2, 2), 'pool1')
    res1 = mp.forward(x)
    print res1.shape

    import lasagne
    import os
    os.environ["THEANO_FLAGS"] = "device=cpu"
    import theano
    import theano.tensor as T
    theano.config.optimizer='None'
    input_var = T.tensor4('inputs', dtype=theano.config.floatX)
    network = lasagne.layers.InputLayer(shape=(None, ic, 4, 4),
                                        input_var=input_var)
    a = np.arange(0, c).reshape([1, c])
    b = np.arange(0, -c, -1).reshape([c, 1])
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    out = lasagne.layers.get_output(network, deterministic=True)
    #lasagne.layers.set_all_param_values(network, cf)
    fn = theano.function([input_var], [out])
    res2 = fn(x)
    res2 = np.asarray(res2, dtype=np.float32).reshape(res1.shape)

    print "comp MaxPooling with lasagne", np.allclose(res1, res2)
    print res1.shape, res1
    print res2.shape, res2





if __name__ == "__main__":
    # test1()
    # test2()
    # test3()
    # test4()
    #test5()
    #test6()
    test7()
