import numpy as np

from scipy.linalg import circulant

def circulant_check():
    for N in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
      r = np.random.random(N)
      x = np.random.random(N)
      d = np.fft.ifft( np.fft.fft(r) * np.fft.fft(x) ) - np.dot(circulant(r), x)
      print N, np.mean(np.abs(d)), np.linalg.norm(d)
    for N in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
      r = np.random.random(N)
      x = np.random.random([N/4, N])
      d = np.fft.ifft( np.fft.fft(x) * np.fft.fft(r) ).T - np.dot(circulant(r), x.T)
      print N, np.mean(np.abs(d)), np.linalg.norm(d)


class CircFC(object):
    def __init__(self, I, H, name=None):
        # I : input size
        # H : hidden size
        self.k = max(I, H) # r is now padded
        self.r = np.random.random( self.k )
        self.dr = np.zeros(self.k)
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
            new_x = np.pad(x, [(0,0), (0, self.k-x.shape[1])], 'constant', constant_values=0)
            a = np.fft.ifft( np.fft.fft(new_x) * np.fft.fft(self.r) )[:,:self.H]
        else:
            a = np.fft.ifft( np.fft.fft(x) * np.fft.fft(self.r) )[:, :self.H]
        if __debug__:
            if self.k > x.shape[1]:
                new_x = np.pad(x, [(0,0), (0, self.k-x.shape[1])], 'constant', constant_values=0)
                check_a = np.dot(new_x, circulant(self.r).T)[:, :self.H]
            else:
                check_a = np.dot(x, circulant(self.r).T)[:, :self.H]
            assert np.linalg.norm( a - check_a ) < 1e-9
            print self.name, "forward", x.shape, "to", a.shape
        return np.real(a)

    def backward(self, x, d_a):
        if self.k > x.shape[1]:
            # pad data, don't forget to change form of x
            assert self.k == self.H
            new_x = np.pad(x, [(0,0), (0, self.k-x.shape[1])], 'constant', constant_values=0)[:, self.mapping]
            self.dr = np.sum( np.fft.ifft(np.fft.fft(new_x)*np.fft.fft(d_a)), axis=0 )
            d_x = np.fft.ifft( np.fft.fft(d_a) * np.fft.fft(self.r[self.mapping]) )[:,:x.shape[1]]
        else:
            new_x = x[:, self.mapping]
            new_d_a = np.pad(d_a, [(0,0), (0, self.k-d_a.shape[1])], 'constant', constant_values=0)
            self.dr = np.sum( np.fft.ifft(np.fft.fft(new_x)*np.fft.fft(new_d_a)), axis=0 )
            d_x = np.fft.ifft( np.fft.fft(new_d_a) * np.fft.fft(self.r[self.mapping]) )[:,:x.shape[1]]
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



class FC(object):
    def __init__(self, I, H, name=None):
        # I : input size
        # H : hidden size
        self.w = np.random.random([H, I])
        self.dw = np.zeros(self.w.shape)
        self.c = np.random.random([H, 1])
        self.dc = np.zeros(self.c.shape)

        self.name = name

    def forward(self, x):
        a = np.dot(x, self.w.T) + self.c.T # a = wx + c
        if __debug__:
            print self.name, "forward", x.shape, "to", a.shape
        return a

    def backward(self, x, d_a):
        self.dw = np.dot(d_a.T, x)
        self.dc = np.dot(d_a.T, np.ones([x.shape[0], 1]))
        d_x = np.dot(d_a, self.w) # this is for next backpropagate layer
        if __debug__:
            print self.name, "backward", d_a.shape, "to", d_x.shape
        return d_x

    def update(self, lr=0.01):
        self.w = self.w - lr * self.dw
        self.dw.fill(0.)
        self.c = self.c - lr * self.dc
        self.dc.fill(0.)


sigmoid = lambda x: 1. / (1. + np.exp(-x))


class Sigmoid(object):
    def __init__(self, name=None):
        self.name = name

    def forward(self, a):
        b = sigmoid(a)
        if __debug__:
            print self.name, "forward", a.shape, "to", b.shape
        return b

    def backward(self, a, d_b):
        d_a = d_b * sigmoid(a) * (1 - sigmoid(a)) # siga for sigmoid(a)
        if __debug__:
            print self.name, "backward", d_b.shape, "to", d_a.shape
        return d_a

    def update(self):
        pass


class EuclideanLoss(object):

    def forward(self, pred, target):
        cost = 0.5 * np.sum(((pred - target)**2))
        if __debug__:
            print "euclidean loss", cost
        return cost

    def backward(self, pred, target):
        assert pred.shape == target.shape
        d_b = pred - target
        if __debug__:
            print "euclidean loss backward", d_b.shape
        return d_b


def GradientChecking1():
    # this is to just check network can go both forward and backward
    B = 3  # batch size
    I = 5  # input size
    H = 11 # hidden size
    O = 1   # output size

    x = np.random.random([B, I])
    y = np.sum(np.sin(x), axis=1).reshape(B, O)

    layers = [FC(I, H, "fc1"), Sigmoid("sig1"), FC(H, O, "fc2"), Sigmoid("sig2")]
    nlayers = len(layers)

    # forward and backward

    # inputs[i] is the input for i-th layer
    # the last of inputs[i] must be the output of current network
    inputs = [x]
    for i in xrange(nlayers):
        inputs.append( layers[i].forward(inputs[-1]) ) # inputs[i] is the input for i-th layer

    cost = EuclideanLoss()
    # loss = cost.forward(inputs[-1], y)

    # grads[i] is the gradients for i-th layer, but in the reverse order
    grads = [cost.backward(inputs[-1], y)]
    for i in reversed(xrange(nlayers)):
        grads.append( layers[i].backward(inputs[i], grads[-1]) ) # grads[i]

    for l in layers:
        l.update()

    print "Successfully go forward and backward through all layers"


def fwd(x, y, layers, cost):
    inputs = [x]
    nlayers = len(layers)
    for i in xrange(nlayers):
        inputs.append( layers[i].forward(inputs[-1]) ) # inputs[i] is the input for i-th layer
    loss = cost.forward(inputs[-1], y)
    return loss


def GradientChecking2():
    # this is to check fully connected layer
    B = 3  # batch size
    I = 7  # input size
    O = I   # output size

    x = np.random.random([B, I])
    y = np.sin(x)
    #y = np.sum(np.sin(x), axis=1).reshape(B, O)

    layers = [FC(I, O, "fc1")]
    nlayers = len(layers)

    # forward and backward

    # inputs[i] is the input for i-th layer
    # the last of inputs[i] must be the output of current network
    inputs = [x]
    for i in xrange(nlayers):
        inputs.append( layers[i].forward(inputs[-1]) ) # inputs[i] is the input for i-th layer

    cost = EuclideanLoss()
    # loss = cost.forward(inputs[-1], y)

    # grads[i] is the gradients for i-th layer, but in the reverse order
    grads = [cost.backward(inputs[-1], y)]
    for i in reversed(xrange(nlayers)):
        grads.append( layers[i].backward(inputs[i], grads[-1]) ) # grads[i]

    # following checking method is from https://gist.github.com/karpathy/587454dc0146a6ae21fc
    delta = 1e-5
    rel_error_thr_warning = 1e-2
    rel_error_thr_error = 1

    checklist = [layers[0].w, layers[0].c]
    grads_analytic = [layers[0].dw, layers[0].dc]
    names = ['w', 'c']
    for j in xrange(len(checklist)):
        mat = checklist[j]
        dmat = grads_analytic[j]
        name = names[j]
        for i in xrange(mat.size):
            old_val = mat.flat[i]

            # test f(x + delta_x)
            mat.flat[i] = old_val + delta
            loss0 = fwd(x, y, layers, cost)

            # test f(x - delta_x)
            mat.flat[i] = old_val - delta
            loss1 = fwd(x, y, layers, cost)

            mat.flat[i] = old_val # recover

            grad_analytic = dmat.flat[i]
            grad_numerical = (loss0 - loss1) / (2 * delta)

            if grad_numerical == 0 and grad_analytic == 0:
                rel_error = 0 # both are zero, OK.
                status = 'OK'
            elif abs(grad_numerical) < 1e-7 and abs(grad_analytic) < 1e-7:
                rel_error = 0 # not enough precision to check this
                status = 'VAL SMALL WARNING'
            else:
                rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
                status = 'OK'
                if rel_error > rel_error_thr_warning: status = 'WARNING'
                if rel_error > rel_error_thr_error: status = '!!!DANGEROUS ERROR!!!'

            print '%s checking param %s index %s (val = %+8f), analytic = %+8f, numerical = %+8f, relative error = %+8f' \
                    % (status, name, `np.unravel_index(i, mat.shape)`, old_val, grad_analytic, grad_numerical, rel_error)

    print "Finish checking fully connected"

def GradientChecking3():
    # this is to check fully connected layer
    B = 3  # batch size
    I = 7  # input size
    H = 19
    O = I   # output size

    x = np.random.random([B, I])
    y = np.sin(x)
    #y = np.sum(np.sin(x), axis=1).reshape(B, O)

    layers = [CircFC(I, H, "CircFc1"), CircFC(H, O, "CircFc2")]
    nlayers = len(layers)

    # forward and backward

    # inputs[i] is the input for i-th layer
    # the last of inputs[i] must be the output of current network
    inputs = [x]
    for i in xrange(nlayers):
        inputs.append( layers[i].forward(inputs[-1]) ) # inputs[i] is the input for i-th layer

    cost = EuclideanLoss()
    # loss = cost.forward(inputs[-1], y)

    # grads[i] is the gradients for i-th layer, but in the reverse order
    grads = [cost.backward(inputs[-1], y)]
    for i in reversed(xrange(nlayers)):
        grads.append( layers[i].backward(inputs[i], grads[-1]) ) # grads[i]

    # following checking method is from https://gist.github.com/karpathy/587454dc0146a6ae21fc
    delta = 1e-5
    rel_error_thr_warning = 1e-2
    rel_error_thr_error = 1

    checklist = [layers[0].r, layers[1].r]
    grads_analytic = [layers[0].dr, layers[1].dr]
    names = ['r0', 'r1']
        for j in xrange(len(checklist)):
        mat = checklist[j]
        dmat = grads_analytic[j]
        name = names[j]
        for i in xrange(mat.size):
            old_val = mat.flat[i]

            # test f(x + delta_x)
            mat.flat[i] = old_val + delta
            loss0 = fwd(x, y, layers, cost)

            # test f(x - delta_x)
            mat.flat[i] = old_val - delta
            loss1 = fwd(x, y, layers, cost)

            mat.flat[i] = old_val # recover

            grad_analytic = dmat.flat[i]
            grad_numerical = (loss0 - loss1) / (2 * delta)

            if grad_numerical == 0 and grad_analytic == 0:
                rel_error = 0 # both are zero, OK.
                status = 'OK'
            elif abs(grad_numerical) < 1e-7 and abs(grad_analytic) < 1e-7:
                rel_error = 0 # not enough precision to check this
                status = 'VAL SMALL WARNING'
            else:
                rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
                status = 'OK'
                if rel_error > rel_error_thr_warning: status = 'WARNING'
                if rel_error > rel_error_thr_error: status = '!!!DANGEROUS ERROR!!!'

            print '%s checking param %s index %s (val = %+8f), analytic = %+8f, numerical = %+8f, relative error = %+8f' \
                    % (status, name, `np.unravel_index(i, mat.shape)`, old_val, grad_analytic, grad_numerical, rel_error)

    print "Finish checking fully connected"



if __name__ == "__main__":
    #GradientChecking1()
    #GradientChecking2()
    #circulant_check()
    GradientChecking3()
