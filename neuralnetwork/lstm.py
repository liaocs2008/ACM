import numpy as np

from nn import FC, Sigmoid, EuclideanLoss


sigmoid = lambda x: 1. / (1. + np.exp(-x))


class LSTMLayer(object):

    def __init__(self, I, K, name=None):
        # I : Input size
        # K : Hidden size
        self.wx = np.random.random([1+I, 4*K]) # [wI, wF, wO, wC]
        self.d_wx = np.zeros(self.wx.shape)
        self.wh = np.random.random([K, 4*K])
        self.d_wh = np.zeros(self.wh.shape)
        self.name = name

    def forward_t(self, x, sc0=None, h0=None):
        # forward at time step t
        # x : input at time step t
        # sc0, h0 are results from last time step
        K = self.wh.shape[1] / 4
        state = np.zeros([x.shape[0], 6*K]) # [bI, bF, bO, g(aC), sc, h]

        # [aI, aF, aO, aC]
        state[:, :4*K] = np.dot(np.concatenate((x, np.ones([x.shape[0], 1])), axis=1), self.wx)
        if h0 is not None:
            state[:, :4*K] += np.dot(h0, self.wh)
        # [bI, bF, bO]
        state[:, :3*K] = sigmoid(state[:, :3*K])
        # [g(aC)]
        state[:, 3*K:4*K] = np.tanh(state[:, 3*K:4*K])
        # sc
        state[:, 4*K:5*K] = state[:, :1*K] * state[:, 3*K:4*K]
        if sc0 is not None:
            state[:, 4*K:5*K] += state[:, 1*K:2*K] * sc0
        # h
        state[:, 5*K:6*K] = state[:, 2*K:3*K] * np.tanh(state[:, 4*K:5*K])

        return state

    def forward(self, x, sc_init=None, h0=None):
        # x : a list of inputs over time steps
        # sc0 : initial state
        # h0 : initial result, which should be None at first time
        k = self.wh.shape[1] / 4
        T = len(x)
        state = []
        sc0 = sc_init
        for t in xrange(T):
            state.append( self.forward_t(x[t], sc0, h0) )
            sc0 = state[-1][:, 4*k:5*k]
            h0 = state[-1][:, 5*k:6*k]
        return state

    def backward_t(self, d_sc, d_h, x, state, sc0=None):
        # d_sc : do / dsc at time step t
        # d_h : do / dh at time step t
        # state : state at time step t
        # sc0 : sc at previous time step t-1
        # return d_sc, d_h, d_x of next backpropagation
        K = self.wh.shape[1] / 4
        grad = np.zeros([state.shape[0], 4*K]) # [d_aI, d_aF, d_aO, d_aC]
        # d_aO
        grad[:, 2*K:3*K] = d_h * np.tanh(d_h) * state[:, 2*K:3*K] * (1. - state[:, 2*K:3*K])
        # d_sc
        d_sc += d_h * state[:, 2*K:3*K] * (1. - np.tanh(state[:, 4*K:5*K])**2)
        # d_aC
        grad[:, 3*K:4*K] = d_sc * state[:, :1*K] * (1 - state[:, 3*K:4*K]**2)
        # d_aF
        if sc0 is not None:
            grad[:, 1*K:2*K] = d_sc * sc0 * state[:, 1*K:2*K] * (1. - state[:, 1*K:2*K])
        # d_aI
        grad[:, :1*K] = d_sc * state[:, 3*K:4*K] * state[:, :1*K] * (1. - state[:, :1*K])
        # d_w
        self.d_wx += np.dot(np.concatenate((x, np.ones([x.shape[0], 1])), axis=1).T, grad)
        self.d_wh += np.dot(state[:,5*K:6*K].T, grad)
        return d_sc * state[:, 4*K:5*K], np.dot(grad, self.wh.T), np.dot(grad, self.wx[1:,:].T)

    def backward(self, x, state, d_h_o, sc_init=None):
        # d_h_o : a list of do/dht
        k = self.wh.shape[1] / 4
        T = len(d_h_o)
        d_x = []
        d_h = np.copy(d_h_o)
        d_sc = np.zeros([state[0].shape[0], k])
        for t in reversed(xrange(T)):
            if 0 == t:
                sc0 = sc_init
            else:
                sc0 = state[t-1][:, 4*k:5*k]
            d_sc, d_h_tmp, d_x_tmp = self.backward_t(d_sc, d_h[t], x[t], state[t], sc0)
            d_h[t-1] += d_h_tmp
            d_x.append(d_x_tmp)
        return d_x, d_h

    def update(self, lr=0.01):
        self.wx = self.wx - lr * self.d_wx
        self.d_wx.fill(0.)
        self.wh = self.wh - lr * self.d_wh
        self.d_wh.fill(0.)


def checkSequentialMatchesBatch():
    # this is to just check network can go both forward and backward
    B = 3   # batch size
    I = 19  # input size
    K = 11  # hidden size
    K2 = 13 #
    O = I   # output size
    T = 4

    x = []
    y = []
    for t in xrange(T):
        x.append( np.random.random([B, I]) * 2 * np.pi / T + float(t) * 2 * np.pi / T )
        y.append( np.sin(x[-1]) )

    fc1 = FC(I, K, "fc1")
    lstm = LSTMLayer(K, K2, "lstm1")
    fc2 = FC(K2, O, "fc2")
    cost = EuclideanLoss()

    pred = [] # predictions
    # forward checking

    inputs = [fc1.forward(xi) for xi in x]

    # one time forward
    state = lstm.forward(inputs)

    # sequential forward
    sc0 = None
    h0 = None
    for t in xrange(T):
        st = lstm.forward_t(inputs[t], sc0, h0)
        sc0 = st[:, 4*K2:5*K2]
        h0 = st[:, 5*K2:6*K2]
        assert np.allclose(st, state[t])
        pred.append( fc2.forward(h0) )

    grads = []
    for t in xrange(T):
        loss = cost.forward(pred[t], y[t])
        grads.append( fc2.backward( state[t][:, 5*K2:6*K2], cost.backward(pred[t], y[t]) ) )

    # backward checking

    # one time backward
    d_x, d_h = lstm.backward(inputs, state, grads)

    # sequential backward
    d_sc = np.zeros([state[0].shape[0], K2])
    for t in reversed(xrange(T)):
        if 0 == t:
            sc0 = None
        else:
            sc0 = state[t-1][:, 4*K2:5*K2]
        d_sc, d_ht, d_xt = lstm.backward_t(d_sc, grads[t], inputs[t], state[t], sc0)
        grads[t-1] += d_ht # here change the value of grads
        assert np.allclose(d_xt, d_x[t])
        assert np.allclose(d_h[t], grads[t])
        fc1.backward(x[t], d_x[t]) # here should apply d_x from lstm rather than grads from fc2

    print "Successfully go forward and backward through all layers"





def fwd(x, y, lstm, cost):
    inputs = [x]
    nlayers = len(layers)
    for i in xrange(nlayers):
        inputs.append( layers[i].forward(inputs[-1]) ) # inputs[i] is the input for i-th layer
    loss = cost.forward(inputs[-1], y)
    return loss

def GradientChecking():
    # this is to check lstm layer
    B = 3  # batch size
    I = 7  # input size
    O = I   # output size
    T = 4

    x = []
    y = []
    for t in xrange(T):
        x.append( np.random.random([B, I]) * 2 * np.pi / T + float(t) * 2 * np.pi / T )
        y.append( np.sin(x[-1]) )

    lstm = LSTMLayer(I, O, "lstm1")

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

    checklist = [layers[0].w]
    grads_analytic = [layers[0].dw]
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


if __name__ == "__main__":
    checkSequentialMatchesBatch()