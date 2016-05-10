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

        # I + K + 1, 4 * hidden_size
        w = [[ -1.66093506e-01,   3.60780883e-01,   1.77400450e-01,
              2.88065733e-01,  -2.88711177e-01,  -3.27027922e-01,
              2.94834957e-01,   1.70812918e-01,  -5.97370622e-02,
             -1.47481651e-01,  -3.76395556e-02,  -1.93776949e-01,
              3.38224151e-02,  -2.16094809e-01,  -7.60638526e-02,
             -4.81194577e-02],
           [  3.21665278e-01,  -3.59514409e-02,  -3.39971043e-01,
             -1.46632129e-01,  -5.60674411e-01,   2.60440242e-03,
              3.54662102e-01,   2.44007439e-01,  -9.68212604e-02,
             -1.53042166e-01,   1.00619831e-01,  -5.22637829e-01,
             -1.09055054e-01,   3.26302375e-02,  -2.94504845e-01,
              1.48039930e-01],
           [  3.74181313e-02,  -3.42388286e-02,   1.98701941e-01,
             -2.83676479e-01,   2.72996099e-02,  -2.44554330e-02,
              2.69382111e-01,   3.07712825e-01,  -2.41987821e-01,
             -1.51019976e-01,  -1.71844193e-01,   3.07666541e-01,
             -1.66065493e-01,   1.53635389e-01,   1.51132737e-02,
              3.92110213e-01],
           [ -2.44767698e-01,   2.22862604e-01,  -4.97412601e-02,
              1.28887742e-01,  -4.13419698e-02,   1.38312622e-02,
             -2.91197868e-01,   2.47479349e-01,  -1.22203487e-01,
              1.35145017e-01,  -1.60531708e-01,   2.14666929e-01,
              2.74342362e-01,   1.83679518e-01,   1.94361799e-01,
              2.08341318e-03],
           [ -1.85405815e-01,   5.90825996e-01,  -1.34555475e-01,
             -8.60998643e-02,   2.69033552e-01,   1.92147266e-01,
             -1.41197735e-01,  -1.74699022e-01,   4.31336918e-02,
             -7.03833999e-02,   2.28511273e-01,  -1.72850813e-02,
             -5.03143527e-01,  -4.25644426e-01,   4.30959999e-02,
              1.26090277e-01],
           [  1.98195520e-01,  -1.39990670e-01,  -1.18927715e-01,
             -2.93643977e-01,   6.01610074e-02,   2.47611081e-02,
              6.66888891e-01,  -4.04114757e-01,  -1.07643397e-01,
              7.83862881e-02,   3.75513708e-01,  -4.32352572e-02,
              2.49357237e-01,   4.91738393e-04,   6.47379018e-02,
              2.72836324e-01],
           [  6.49655948e-02,   3.25800018e-01,   3.04403969e-01,
             -4.41494466e-01,  -4.56808747e-02,  -4.68771386e-01,
              1.84827336e-01,  -1.37334789e-01,   5.47315421e-01,
             -2.20676594e-01,   5.90917213e-02,  -1.88829265e-01,
             -1.17142229e-01,   6.42669407e-01,   3.32808028e-01,
             -4.60222307e-02],
           [ -9.94672048e-02,  -9.98579311e-02,   1.10521682e-01,
             -5.62008182e-02,   9.26530385e-02,   1.09434038e-01,
             -9.78510672e-02,  -1.42209633e-01,   3.90758104e-01,
              1.45180443e-01,   1.38895523e-02,   2.78230739e-01,
             -1.55566156e-02,  -1.63514033e-01,   1.05226241e-01,
             -2.14829739e-01],
           [  3.18655353e-01,  -3.27296215e-01,  -1.18102498e-01,
             -2.41594363e-01,   2.15228370e-01,   2.03116265e-01,
             -5.11606013e-03,  -5.99708692e-02,  -1.32339903e-01,
              8.80670217e-02,  -1.77702402e-02,  -1.17950964e-01,
             -6.42567095e-02,  -1.85071819e-01,  -3.30887944e-01,
              2.02877194e-01],
           [ -5.90697680e-01,   3.88863074e-01,   6.35008620e-02,
              1.12772399e-01,  -5.91835791e-01,  -3.02245934e-01,
              3.86661979e-01,   2.12688416e-01,  -8.02671497e-02,
              2.02404668e-01,  -2.68080219e-01,   1.45781096e-01,
             -5.80895619e-03,  -2.63705160e-01,  -5.02134094e-02,
             -5.05463031e-02],
           [  1.72289875e-01,   3.29517551e-01,  -1.16107070e-01,
              6.12272377e-01,   3.48040404e-02,   6.37133245e-02,
              2.97636758e-01,   6.09493829e-01,  -2.11681454e-01,
              2.47073024e-01,  -1.52354428e-01,   3.34437396e-01,
              1.19143194e-01,  -2.61937484e-01,   2.17918936e-02,
              3.55083305e-01],
           [ -1.84717752e-01,  -3.87994172e-01,   1.16284350e-01,
              2.26118378e-01,  -2.91778694e-01,   7.43330386e-02,
             -2.82710138e-01,   1.17509269e-02,   7.22880329e-02,
              7.89276483e-02,   8.56337730e-04,  -2.87785130e-01,
              5.03478828e-01,  -1.05057826e-02,  -2.81652088e-01,
             -3.44178454e-01],
           [  4.60616923e-01,  -3.09103801e-01,   1.20877650e-01,
              2.64452499e-01,  -1.85510004e-01,  -2.75700227e-01,
             -2.02106445e-01,   2.11200955e-01,   1.60148356e-01,
              6.85189021e-01,   1.74488854e-01,   3.33231089e-01,
              6.07693612e-01,   6.00491027e-01,  -1.86007320e-01,
              3.70265934e-01],
           [ -1.01443040e-01,  -2.49433365e-01,   9.88694394e-02,
              3.33722294e-01,  -1.41129338e-01,   1.86202291e-01,
              4.33845957e-01,   6.02757000e-02,  -1.95461775e-01,
              1.51793702e-01,  -2.09775031e-01,   1.61092323e-01,
              1.04369289e-01,  -3.32418845e-03,   2.77256854e-01,
              1.52747161e-01],
           [  5.92036111e-02,  -2.05526391e-01,  -2.29726688e-02,
              9.71108999e-02,   5.31397210e-04,   1.53776726e-01,
              5.15856536e-01,   2.31486242e-01,  -1.16097210e-01,
              5.60506981e-01,   1.82814970e-01,   5.08359079e-02,
             -4.69946139e-02,   5.27091115e-01,   1.25761816e-01,
              4.45297605e-01]]
        print "WLSTM=", np.sum(np.abs(w))
        w = np.array(w)
        self.wx = w[:1+I,:]
        self.wx[0,:] = 0. # bias set as 0
        self.wh = w[1+I:,:]

    def forward_t(self, x, sc0=None, h0=None):
        # forward at time step t
        # x : input at time step t
        # sc0, h0 are results from last time step
        K = self.wh.shape[1] / 4
        state = np.zeros([x.shape[0], 6*K]) # [bI, bF, bO, g(aC), sc, h]

        # [aI, aF, aO, aC]
        state[:, :4*K] = np.dot(np.concatenate((np.ones([x.shape[0], 1]), x), axis=1), self.wx)
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

    def forward(self, x, sc_init=None, h_init=None):
        # x : a list of inputs over time steps
        # sc0 : initial state
        # h0 : initial result, which should be None at first time
        k = self.wh.shape[1] / 4
        T = len(x)
        state = []
        sc0 = sc_init
        h0 = h_init
        for t in xrange(T):
            """
            print "t=", t
            print "prevh", np.sum(np.abs(h0))
            print "X[t]", np.sum(np.abs(x[t]))
            print "WLSTM", np.sum(np.abs(self.wx)) + np.sum(np.abs(self.wh))
            """
            state.append( self.forward_t(x[t], sc0, h0) )
            sc0 = state[-1][:, 4*k:5*k]
            h0 = state[-1][:, 5*k:6*k]
            print "t===", t
            print "C[t]", np.sum(np.abs(sc0))
            print "Ct[t]", np.sum(np.abs(np.tanh(sc0)))
            print "Hout[t]", np.sum(np.abs(h0))
        return state

    def backward_t(self, d_sc, d_h, x, state, sc0=None, h0=None):
        # d_sc : do / dsc at time step t
        # d_h : do / dh at time step t
        # state : state at time step t
        # sc0 : sc at previous time step t-1
        # return d_sc, d_h, d_x of next backpropagation
        K = self.wh.shape[1] / 4
        grad = np.zeros([state.shape[0], 4*K]) # [d_aI, d_aF, d_aO, d_aC]
        # d_aO
        grad[:, 2*K:3*K] = d_h * np.tanh(state[:,4*K:5*K]) * state[:, 2*K:3*K] * (1. - state[:, 2*K:3*K])
        print "d_omega=", np.sum(np.abs(np.tanh(state[:,4*K:5*K]))), np.sum(np.abs(d_h))
        # d_sc
        d_sc += d_h * state[:, 2*K:3*K] * (1. - np.tanh(state[:, 4*K:5*K])**2)
        print "dHout[t]=", np.sum(np.abs(d_h))
        print "dC[t]=", np.sum(np.abs(d_sc))
        # d_aC
        grad[:, 3*K:4*K] = d_sc * state[:, :1*K] * (1 - state[:, 3*K:4*K]**2)
        # d_aF
        if sc0 is not None:
            grad[:, 1*K:2*K] = d_sc * sc0 * state[:, 1*K:2*K] * (1. - state[:, 1*K:2*K])
        # d_aI
        grad[:, :1*K] = d_sc * state[:, 3*K:4*K] * state[:, :1*K] * (1. - state[:, :1*K])



        if sc0 is not None: print "dIFOGft_p=", np.sum(np.abs(d_sc * sc0))
        print "dIFOGft_i=", np.sum(np.abs(d_sc * state[:, 3*K:4*K]))
        print "dIFOGft_ac=", np.sum(np.abs(d_sc * state[:, :1*K]))
        print "y=", np.sum(np.abs(state[:,:3*K]))
        print "dIFOG_ipw=", np.sum(np.abs(grad[:,:3*K]))
        # d_w
        self.d_wx += np.dot(np.concatenate((np.ones([x.shape[0], 1]), x), axis=1).T, grad)
        if h0 is not None:
            self.d_wh += np.dot(h0.T, grad)
        return d_sc * state[:, 1*K:2*K], np.dot(grad, self.wh.T), np.dot(grad, self.wx[1:,:].T)

    def backward(self, x, state, d_h_o, sc_init=None, h_init=None):
        # d_h_o : a list of do/dht
        k = self.wh.shape[1] / 4
        T = len(d_h_o)
        d_x = []
        d_h = np.copy(d_h_o)
        d_sc = np.zeros([state[0].shape[0], k])
        for t in reversed(xrange(T)):
            if 0 == t:
                sc0 = sc_init
                h0 = h_init
            else:
                sc0 = state[t-1][:, 4*k:5*k]
                h0 = state[t-1][:, 5*k:6*k]
            print "t===", t
            d_sc, d_h_tmp, d_x_tmp = self.backward_t(d_sc, d_h[t], x[t], state[t], sc0, h0)
            print "updated WLSTM=", np.sum(np.abs(self.d_wh))+np.sum(np.abs(self.d_wx))
            print "dHin[t]=", np.sum(np.abs(d_h_tmp)) + np.sum(np.abs(d_x_tmp))
            if 0 != t:
                d_h[t-1] += d_h_tmp
            d_x.append(d_x_tmp)
        return d_x, d_h, d_sc, d_h_tmp # now d_sc = d_sc0, d_h_tmp = d_h0

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

    sc0 = np.random.random([inputs[0].shape[0], K2])
    h0 = np.random.random([inputs[0].shape[0], K2]) # this must relate to its input size
    # one time forward
    state = lstm.forward(inputs, sc0, h0)

    # sequential forward

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
    d_x, d_h, d_sc0, d_h0 = lstm.backward(inputs, state, grads)

    # sequential backward
    d_sc = np.zeros([state[0].shape[0], K2])
    for t in reversed(xrange(T)):
        if 0 == t:
            sc_prev = sc0
            h_prev = h0
        else:
            sc_prev = state[t-1][:, 4*K2:5*K2]
            h_prev = state[t-1][:, 5*K2:6*K2]
        d_sc, d_ht, d_xt = lstm.backward_t(d_sc, grads[t], inputs[t], state[t], sc_prev, h_prev)
        if 0 != t:
            grads[t-1] += d_ht # here change the value of grads
            assert np.allclose(d_xt, d_x[t])
            assert np.allclose(d_h[t], grads[t])
        else:
            assert np.allclose(d_sc, d_sc0)
            assert np.allclose(d_ht, d_h0)
        fc1.backward(x[t], d_x[t]) # here should apply d_x from lstm rather than grads from fc2


    print "Successfully go forward and backward through all layers"


def fwd(x, y, sc0, h0, lstm, cost):
    T = len(x)
    B = x[0].shape[0]
    K = O = y.shape[1]
    state = lstm.forward(x, sc0, h0)
    pred = np.zeros([T*B, O])
    for t in xrange(T):
        pred[t*B:(t+1)*B,:] = state[t][:, 5*K:6*K]
    loss = cost.forward(pred, y)
    return loss


def GradientChecking():
    # this is to check lstm layer
    # euclidean cost happens to have the linearity
    # its gradients is the same even no matter doing in T batches or doing in a single one

    B = 3  # batch size
    I = 7  # input size
    K = O = I   # output size
    T = 4

    #sc0 = np.random.random([B, K])
    #h0 = np.random.random([B, K])
    sc0 = None
    h0 = None

    x = []
    y = np.zeros([T*B, O])
    for t in xrange(T):
        x.append( np.random.random([B, I]) * 2 * np.pi / T + float(t) * 2 * np.pi / T )
        y[t*B:(t+1)*B,:] = np.sin(x[-1])

    lstm = LSTMLayer(I, K, "lstm1")
    cost = EuclideanLoss()

    state = lstm.forward(x, sc0, h0)
    pred = np.zeros([T*B, O])
    for t in xrange(T):
        pred[t*B:(t+1)*B,:] = state[t][:, 5*K:6*K]
    loss = cost.forward(pred, y)

    grads = np.split(cost.backward(pred, y), T, axis=0)
    for t in xrange(T):
        g = cost.backward(pred[t*B:(t+1)*B,:], y[t*B:(t+1)*B,:])
        assert np.allclose(g, grads[t])
    d_x, d_h, d_sc0, d_h0 = lstm.backward(x, state, grads)

    # following checking method is from https://gist.github.com/karpathy/587454dc0146a6ae21fc
    delta = 1e-5
    rel_error_thr_warning = 1e-2
    rel_error_thr_error = 1

    checklist = [lstm.wx, lstm.wh, sc0, h0]
    grads_analytic = [lstm.d_wx, lstm.d_wh, d_sc0, d_h0]
    names = ['wx', 'wh', 'sc0', 'h0']
    for j in xrange(len(checklist)):
        mat = checklist[j]
        if mat is None: continue
        dmat = grads_analytic[j]
        name = names[j]
        for i in xrange(mat.size):
            old_val = mat.flat[i]

            # test f(x + delta_x)
            mat.flat[i] = old_val + delta
            loss0 = fwd(x, y, sc0, h0, lstm, cost)

            # test f(x - delta_x)
            mat.flat[i] = old_val - delta
            loss1 = fwd(x, y, sc0, h0, lstm, cost)

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


def CheckwithKarparthyImplementation():
    B = 3
    K = 4
    T = 5
    I = 10
    X = [[[-0.46053753, -0.12702845, -0.04676861,  1.20541086, -1.22840403,
          0.12823238,  0.25894951,  1.31279056, -0.47367664, -0.77675997],
          [-0.13589622, -0.81225197,  0.46968238,  1.39512178,  1.00071138,
          -1.21961833,  1.36858751, -0.19675016, -0.20529525,  1.19915128],
          [-0.61118392, -0.26522034,  1.3316866 ,  0.23107417, -0.12793613,
          0.1475729 ,  1.81836196,  1.07397226, -0.65774275, -1.16195936]],

          [[-2.08561904, -1.13641926, -0.77696586,  0.05603776,  0.66275558,
          -0.72845745, -0.01441713, -0.55315556, -0.89294131,  0.0175039 ],
          [-0.11225478,  0.30142428,  0.13769522,  0.09881715, -0.33924172,
          -0.96639354, -1.33296022, -1.06325253, -0.35485236, -0.5269167 ],
          [-0.78573577,  0.33489887,  0.53510236, -0.96803994,  1.13376348,
          -1.03438291, -0.20837116,  0.72300426, -0.79099923, -0.80311962]],

          [[ 1.03492579, -0.25478291, -1.17103354, -1.41569601,  1.59742199,
          -1.18924696, -0.97924339, -0.05057663, -0.50202148, -1.34781123],
          [ 0.59111991, -0.34856999, -0.23229246,  0.89069231, -0.29003402,
          -0.57849048, -0.0333561 , -0.49513068,  0.78706835,  0.97788511],
          [-0.4730505 , -0.94981853,  0.34453016, -1.19295797, -1.26725216,
          -1.25180836,  0.90393459,  0.85666898, -0.47542522, -1.37649252]],

          [[-0.27917267,  1.736293  ,  0.91208764,  0.58495547,  0.38184429,
          0.00337824, -0.14664973,  1.29654035,  0.89376606, -0.18570634],
          [ 0.60909807,  1.01957421,  0.04366278,  0.80829198, -1.11094668,
          0.85784295,  0.16427848, -0.37091095,  1.1421337 , -0.38729937],
          [-1.34285138, -0.58710154, -0.2686603 , -0.75556855, -1.05040411,
          -0.66114968,  2.27931595, -0.74640352, -0.34049966,  1.30704973]],

          [[ 0.2125005 ,  0.51996945,  0.28254926,  1.42968727, -2.04139021,
          -1.60936086,  0.86173618,  0.02007461, -0.06620268, -0.92313161],
          [ 0.95908156,  0.11126797, -0.69499882, -0.05511167, -0.65700018,
          1.38099553, -1.32288244,  1.58502516, -0.42858109,  1.1820933 ],
          [ 1.00174275, -0.50641702, -1.34545112, -0.57526256, -0.40813825,
          -1.71143128,  1.51557555, -0.15594834,  1.56832722,  0.83709094]]]

    h0 = [[-0.33895358, -0.69210718, -0.0312983 , -0.32228987],
             [ 0.44902875, -0.99363984,  0.16780914,  2.01856807],
                    [-0.81207382, -1.53154913, -1.25699384, -1.68362838]]

    c0 = [[-1.89859351, -2.4162538 ,  0.42149549, -0.04808366],
             [-0.56061951, -1.07429506, -0.92592173, -1.63058523],
                    [-0.25688874,  0.41776216, -0.31470688,  0.78955328]]

    lstm = LSTMLayer(I, K, "lstm1")

    X1 = np.split(np.array(X).reshape(T,B,I), T, axis=0)
    X = [x.reshape(B,I) for x in X1]
    h0 = np.array(h0)
    sc0 = np.array(c0)

    state = lstm.forward(X, sc0, h0)
    H = np.zeros([T,B,K])
    for t in xrange(T):
        H[t,:,:] = state[t][:,5*K:6*K]
    print "H=", np.sum(np.abs(H))

    wrand = [[[-2.19645915, -0.20372572,  0.04667188,  1.1777223 ],
        [ 0.51888014,  0.30890112,  0.07520759, -0.6635596 ],
        [ 1.60175509, -0.90490336, -0.20282142, -1.4610382 ]],

       [[ 1.22886321, -0.64816003, -0.04335894,  0.40218784],
        [-0.62672513, -1.47956223,  1.53894697,  1.19967836],
        [-0.39698541, -0.89571055, -2.03079263,  0.4002015 ]],

       [[ 0.28975391,  0.69233427, -0.19667373,  2.36946528],
        [ 0.09754283, -1.29803812,  1.80872805, -0.31760962],
        [ 1.09012875, -1.52293371,  0.85862186,  0.83709996]],

       [[-2.86371197, -0.56151666,  0.86346862,  0.80527104],
        [ 0.08599282,  0.45209014, -0.54929409,  1.55889094],
        [-0.79173911,  0.52392564,  0.30710522,  0.20023893]],

       [[ 0.2993529 , -0.07263059,  0.41278694, -0.64327391],
        [-0.18974562,  0.53875572,  1.59889773, -0.55106615],
        [ 0.84458987, -0.11670773,  1.02139639,  0.82269492]]]
    wrand = np.array(wrand)
    loss = np.sum(H * wrand)
    print "loss, ", loss
    print "wrand", np.sum(np.abs(wrand))
    dH = np.split(wrand, T, axis=0)
    grads = [d_h.reshape(B, K) for d_h in dH]

    d_x, d_h, d_sc0, d_h0 = lstm.backward(X, state, grads, sc0, h0)
    print "BdX", np.sum(np.abs(np.array(d_x)))
    print "BdWLSTM", np.sum(np.abs(lstm.d_wh))+np.sum(np.abs(lstm.d_wx))
    print "Bdc0", np.sum(np.abs(d_sc0))
    print "Bdh0", np.sum(np.abs(d_h0))



if __name__ == "__main__":
    #checkSequentialMatchesBatch()
    #GradientChecking()
    CheckwithKarparthyImplementation()