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
        self.dw = 0 # this is to clear, corresponding to the accumulation in backward
        w = self.w.ravel()[::-1].reshape([self.I, self.I])
        s = [self.I + x.shape[0] - 1, self.I + x.shape[1] - 1]
        r = np.fft.irfft2(np.fft.rfft2(w, s) * np.fft.rfft2(x, s), s)[(self.I-1):(x.shape[0]), (self.I-1):(x.shape[1])]
        if __debug__:
            fr = np.zeros([x.shape[0] - self.w.shape[0] + 1, x.shape[1] - self.w.shape[1] + 1])
            for i in xrange(r.shape[0]):
                for j in xrange(r.shape[1]):
                    fr[i,j] = np.sum( self.w * x[i:i+self.w.shape[0],j:j+self.w.shape[1]] )
            assert np.allclose(fr, r)
            print self.name, "forward", x.shape, "to", r.shape
        return r

    def backward(self, x, d_a):
        da = d_a.ravel()[::-1].reshape(d_a.shape)
        s = [da.shape[0] + x.shape[0] - 1, da.shape[1] + x.shape[1] - 1]
        dw = np.fft.irfft2(np.fft.rfft2(da, s) * np.fft.rfft2(x, s), s)
        dw = dw[(d_a.shape[0]-1):(x.shape[0]), (d_a.shape[1]-1):(x.shape[1])]
        self.dw += dw # !!!!!! notice that this is for multi-channel

        s = [d_a.shape[0] + self.I - 1, d_a.shape[1] + self.I - 1]
        d_x = np.fft.irfft2(np.fft.rfft2(d_a, s) * np.fft.rfft2(self.w, s), s)

        if __debug__:
            fdw = np.zeros(self.dw.shape)
            for a in xrange(self.w.shape[0]):
                for b in xrange(self.w.shape[1]):
                    fdw[a,b] = np.sum(d_a * x[a:a+d_a.shape[0],b:b+d_a.shape[1]])
            assert np.allclose(fdw, dw)

            fdx = np.zeros(x.shape) # this is for next backpropagate layer
            assert d_a.shape[0] + self.w.shape[0] - 1 == d_x.shape[0]
            assert d_a.shape[1] + self.w.shape[1] - 1 == d_x.shape[1]
            for i in xrange(d_a.shape[0]):
                for j in xrange(d_a.shape[1]):
                    for a in xrange(self.w.shape[0]):
                        for b in xrange(self.w.shape[1]):
                            fdx[i+a,j+b] += d_a[i,j] * self.w[a, b]
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
        self.b = [[BaseConv(filter_size, (name + '_tmp_i%d_o%d' % (i, j))) for i in xrange(self.c)] for j in xrange(self.f)]
        self.w = oW
        self.name = name

    def forward(self, x):
        a = np.zeros([x.shape[0], self.f, self.w, self.w])
        for i in xrange(a.shape[0]):
            for f in xrange(self.f):
                for c in xrange(self.c):
                    a[i,f,:,:] += self.b[f][c].forward(x[i,c,:,:])
        if __debug__:
            print self.name, "forward", x.shape, "to", a.shape
        return a

    def backward(self, x, d_a):
        d_x = np.zeros(x.shape)
        for i in xrange(d_a.shape[0]):
            for f in xrange(self.f):
                for c in xrange(self.c):
                    d_x[i,c,:,:] += self.b[f][c].backward(x[i,c,:,:], d_a[i,f,:,:])
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
    f = np.random.random([c, w1*h1])

    a = np.arange(0,c).reshape([1,c])
    b = np.arange(0,-c,-1).reshape([c,1])
    #print a+b
    ind = a + b
    #print f
    #print "==="
    cf = f[ind,:].reshape(c, c*w1*h1).T
    #print cf

    indx = np.zeros(c*w1*h1,dtype=np.int)
    for i in xrange(w1*h1):
        indx[i*c:(i+1)*c] = i + np.arange(0,c*w1*h1,w1*h1)
    #print indx
    print cf[indx,:]






def im2col_sliding_broadcasting(A, BSZ, stepsize=1):
    # Parameters
    M,N = A.shape
    col_extent = N - BSZ[1] + 1
    row_extent = M - BSZ[0] + 1
    # Get Starting block indices
    start_idx = np.arange(BSZ[0])[:,None]*N + np.arange(BSZ[1])
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)
    # Get all actual indices & index into input array for final output
    return np.take (A,start_idx.ravel()[:,None] + offset_idx.ravel()[::stepsize])


def leo_im2col(x, (w1, h1)):
    # don't support stride and padding now
    batch, c0, w0, h0 = x.shape
    w2 = w0 - w1 + 1
    h2 = h0 - h1 + 1
    a = np.zeros([c0*w1*h1, batch*w2*h2])
    for b in xrange(batch):
        for c in xrange(c0):
            a[c*w1*h1:(c+1)*w1*h1, b*w2*h2:(b+1)*w2*h2] = im2col_sliding_broadcasting(x[b, c, :, :], [w1, h1])
    return a.T


def test2():
    x = np.random.random([3, 7, 7])
    ind = leo_im2col(x, [3, 3])
    print ind.shape, x.shape








def init_w(shape):
    return np.random.normal(0, 1. / np.product(shape), shape)



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
            self.indx[i * self.c:(i + 1) * self.c] = i + np.arange(0, self.c * self.filter_size * self.filter_size, self.filter_size * self.filter_size)
        a = np.arange(0, self.c).reshape([1, self.c])
        b = np.arange(0, -self.c, -1).reshape([self.c, 1])
        self.indx2 = a + b
        self.name = name

    def forward(self, x):
        self.cf = self.f[self.indx2, :].reshape(self.c, self.c * self.filter_size * self.filter_size).T
        self.cf = self.cf[self.indx, :]
        new_x = leo_im2col(x, (self.filter_size, self.filter_size))
        assert new_x.shape == (x.shape[0]*self.oW*self.oD, self.filter_size*self.filter_size*self.iC)
        new_x = np.pad(new_x, [(0,0), (0, (self.c-self.iC)*self.filter_size*self.filter_size)], 'constant', constant_values=(0, 0))
        a = np.dot(new_x[:,self.indx], self.cf)
        if __debug__:
            print self.name, "forward", x.shape, "to", a.shape
        a = a.reshape([x.shape[0], self.oW*self.oD, self.c])
        a = a.transpose([0, 2, 1]).reshape([x.shape[0], self.c, self.oW, self.oD])
        return a[:, :self.oC, :]

    def backward(self, x, d_a):
        d_x = np.zeros(x.shape)

        return d_x

    def update(self, lr=0.01):
        pass



def test4():
    ic = 2
    oc = 3
    batch = 2
    #assert ic == oc
    x = np.random.random([batch, ic, 4, 4])
    conv = Conv(x.shape, (batch, oc, 2, 2), 'conv0')

    c = np.max([ic, oc])
    f = np.random.random([c, 3, 3])

    a = np.arange(0,c).reshape([1,c])
    b = np.arange(0,-c,-1).reshape([c,1])
    cf = f[a+b, :, :]
    for i in xrange(oc):
        for j in xrange(ic):
            conv.b[i][j].w = cf[i,j,:,:]
    res0 = conv.forward(x)
    print res0.shape

    blockcircconv = BlockCircConv(x.shape, (batch, oc, 2, 2), 'conv1')
    blockcircconv.f = f
    res1 = blockcircconv.forward(x)
    print res1.shape

    print np.allclose(res0, res1)


if __name__ == "__main__":
    #test1()
    #test2()
    test3()
    #test4()
