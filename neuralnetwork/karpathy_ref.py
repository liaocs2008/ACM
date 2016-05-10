# following code from
# https://gist.github.com/karpathy/587454dc0146a6ae21fc

"""
This is a batched LSTM forward and backward pass
"""
import numpy as np
import code

class LSTM:

  @staticmethod
  def init(input_size, hidden_size, fancy_forget_bias_init = 3):
    """
    Initialize parameters of the LSTM (both weights and biases in one matrix)
    One might way to have a positive fancy_forget_bias_init number (e.g. maybe even up to 5, in some papers)
    """
    # +1 for the biases, which will be the first row of WLSTM
    WLSTM = np.random.randn(input_size + hidden_size + 1, 4 * hidden_size) / np.sqrt(input_size + hidden_size)
    WLSTM = [[ -1.66093506e-01,   3.60780883e-01,   1.77400450e-01,
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
    WLSTM = np.array(WLSTM)
    WLSTM[0,:] = 0 # initialize biases to zero
    #if fancy_forget_bias_init != 0:
    #  # forget gates get little bit negative bias initially to encourage them to be turned off
    #  # remember that due to Xavier initialization above, the raw output activations from gates before
    #  # nonlinearity are zero mean and on order of standard deviation ~1
    #  WLSTM[0,hidden_size:2*hidden_size] = fancy_forget_bias_init
    return WLSTM

  @staticmethod
  def forward(X, WLSTM, c0 = None, h0 = None):
    """
    X should be of shape (n,b,input_size), where n = length of sequence, b = batch size
    """
    n,b,input_size = X.shape
    d = WLSTM.shape[1]/4 # hidden size
    if c0 is None: c0 = np.zeros((b,d))
    if h0 is None: h0 = np.zeros((b,d))

    # Perform the LSTM forward pass with X as the input
    xphpb = WLSTM.shape[0] # x plus h plus bias, lol
    Hin = np.zeros((n, b, xphpb)) # input [1, xt, ht-1] to each tick of the LSTM
    Hout = np.zeros((n, b, d)) # hidden representation of the LSTM (gated cell content)
    IFOG = np.zeros((n, b, d * 4)) # input, forget, output, gate (IFOG)
    IFOGf = np.zeros((n, b, d * 4)) # after nonlinearity
    C = np.zeros((n, b, d)) # cell content
    Ct = np.zeros((n, b, d)) # tanh of cell content
    for t in xrange(n):
      # concat [x,h] as input to the LSTM
      prevh = Hout[t-1] if t > 0 else h0
      Hin[t,:,0] = 1 # bias
      Hin[t,:,1:input_size+1] = X[t]
      Hin[t,:,input_size+1:] = prevh
      # compute all gate activations. dots: (most work is this line)
      IFOG[t] = Hin[t].dot(WLSTM)

      """
      print "t=", t
      print "prevh", np.sum(np.abs(prevh))
      print "X[t]", np.sum(np.abs(X[t]))
      print "Hin[t]", np.sum(np.abs(Hin[t]))
      print "WLSTM", np.sum(np.abs(WLSTM))
      print "IFOG[t]", np.sum(np.abs(IFOG[t]))
      """


      # non-linearities
      IFOGf[t,:,:3*d] = 1.0/(1.0+np.exp(-IFOG[t,:,:3*d])) # sigmoids; these are the gates
      IFOGf[t,:,3*d:] = np.tanh(IFOG[t,:,3*d:]) # tanh
      """
      if 0 == t:
        print "IFOGf1", np.sum(np.abs(IFOGf[t,:,:3*d]))
      if 0 == t:
        print "IFOGf2", IFOGf[t,:,3*d:] #np.sum(np.abs(IFOGf[t,:,3*d:]))
      """

      # compute the cell activation
      prevc = C[t-1] if t > 0 else c0
      C[t] = IFOGf[t,:,:d] * IFOGf[t,:,3*d:] + IFOGf[t,:,d:2*d] * prevc
      Ct[t] = np.tanh(C[t])
      Hout[t] = IFOGf[t,:,2*d:3*d] * Ct[t]


      print "t===", t
      print "C[t]", np.sum(np.abs(C[t])) #, np.sum(np.abs(IFOGf[t,:,:d] * IFOGf[t,:,3*d:]))
      print "Ct[t]", np.sum(np.abs(Ct[t]))
      print "Hout[t]", np.sum(np.abs(Hout[t]))





    cache = {}
    cache['WLSTM'] = WLSTM
    cache['Hout'] = Hout
    cache['IFOGf'] = IFOGf
    cache['IFOG'] = IFOG
    cache['C'] = C
    cache['Ct'] = Ct
    cache['Hin'] = Hin
    cache['c0'] = c0
    cache['h0'] = h0

    # return C[t], as well so we can continue LSTM with prev state init if needed
    return Hout, C[t], Hout[t], cache

  @staticmethod
  def backward(dHout_in, cache, dcn = None, dhn = None):

    print "dHout_in", dHout_in.shape
    WLSTM = cache['WLSTM']
    Hout = cache['Hout']
    IFOGf = cache['IFOGf']
    IFOG = cache['IFOG']
    C = cache['C']
    Ct = cache['Ct']
    Hin = cache['Hin']
    c0 = cache['c0']
    h0 = cache['h0']
    n,b,d = Hout.shape
    input_size = WLSTM.shape[0] - d - 1 # -1 due to bias

    # backprop the LSTM
    dIFOG = np.zeros(IFOG.shape)
    dIFOGf = np.zeros(IFOGf.shape)
    dWLSTM = np.zeros(WLSTM.shape)
    dHin = np.zeros(Hin.shape)
    dC = np.zeros(C.shape)
    dX = np.zeros((n,b,input_size))
    dh0 = np.zeros((b, d))
    dc0 = np.zeros((b, d))
    dHout = dHout_in.copy() # make a copy so we don't have any funny side effects
    if dcn is not None: dC[n-1] += dcn.copy() # carry over gradients from later
    if dhn is not None: dHout[n-1] += dhn.copy()
    for t in reversed(xrange(n)):
      print "t=", t

      tanhCt = Ct[t]
      dIFOGf[t,:,2*d:3*d] = tanhCt * dHout[t]
      print "d_omega=", np.sum(np.abs(tanhCt)), np.sum(np.abs(dHout[t]))
      # backprop tanh non-linearity first then continue backprop
      dC[t] += (1-tanhCt**2) * (IFOGf[t,:,2*d:3*d] * dHout[t])

      #print  "tanhCt=", np.sum(np.abs(tanhCt)), np.sum(np.abs(1-tanhCt**2)), np.sum(np.abs(IFOGf[t,:,2*d:3*d]))
      print "dHout[t]=", np.sum(np.abs(dHout[t]))
      print "dC[t]=", np.sum(np.abs(dC[t]))

      if t > 0:
        dIFOGf[t,:,d:2*d] = C[t-1] * dC[t]
        dC[t-1] += IFOGf[t,:,d:2*d] * dC[t]
      else:
        dIFOGf[t,:,d:2*d] = c0 * dC[t]
        dc0 = IFOGf[t,:,d:2*d] * dC[t]
      dIFOGf[t,:,:d] = IFOGf[t,:,3*d:] * dC[t]
      dIFOGf[t,:,3*d:] = IFOGf[t,:,:d] * dC[t]

      print "dIFOGft_p=", np.sum(np.abs(dIFOGf[t,:,d:2*d]))
      print "dIFOGft_i=", np.sum(np.abs(dIFOGf[t,:,:d]))
      print "dIFOGft_ac=", np.sum(np.abs(dIFOGf[t,:,3*d:]))
      # "dIFOGft=", np.sum(np.abs(dIFOGf))

      # backprop activation functions
      dIFOG[t,:,3*d:] = (1 - IFOGf[t,:,3*d:] ** 2) * dIFOGf[t,:,3*d:]
      y = IFOGf[t,:,:3*d]
      dIFOG[t,:,:3*d] = (y*(1.0-y)) * dIFOGf[t,:,:3*d]
      print "y=", np.sum(np.abs(y))
      print "dIFOG_ipw=", np.sum(np.abs(dIFOG[t,:,:3*d])),  np.sum(np.abs(dIFOGf[t,:,:3*d])), np.sum(np.abs(dIFOG[t,:,3*d:]))

      # backprop matrix multiply
      print "dWLSTM=", np.sum(np.abs(dWLSTM)), dWLSTM.shape
      dWLSTM += np.dot(Hin[t].transpose(), dIFOG[t])
      dHin[t] = dIFOG[t].dot(WLSTM.transpose())
      print "updated WLSTM=", np.sum(np.abs(dWLSTM)), dWLSTM.shape
      print "Hin[t].trans()=", np.sum(np.abs(Hin[t].transpose()))
      print "dIFOG[t]=", np.sum(np.abs(dIFOG[t]))
      print "dHin[t]=", np.sum(np.abs(dHin[t]))

      # backprop the identity transforms into Hin
      dX[t] = dHin[t,:,1:input_size+1]
      if t > 0:
        dHout[t-1,:] += dHin[t,:,input_size+1:]
      else:
        dh0 += dHin[t,:,input_size+1:]
      print "dHout=", np.sum(np.abs(dHout))

    return dX, dWLSTM, dc0, dh0



# -------------------
# TEST CASES
# -------------------



def checkSequentialMatchesBatch():
  """ check LSTM I/O forward/backward interactions """

  n,b,d = (5, 3, 4) # sequence length, batch size, hidden size
  input_size = 10
  WLSTM = LSTM.init(input_size, d) # input size, hidden size
  print WLSTM.shape
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

  X = np.array(X)
  h0 = np.array(h0)
  c0 = np.array(c0)
  #X = np.random.randn(n,b,input_size)
  #h0 = np.random.randn(b,d)
  #c0 = np.random.randn(b,d)

  # sequential forward
  cprev = c0
  hprev = h0
  caches = [{} for t in xrange(n)]
  Hcat = np.zeros((n,b,d))
  for t in xrange(n):
    xt = X[t:t+1]
    """
    print "================="
    print "cprev", np.sum(np.abs(cprev))
    print "hprev", np.sum(np.abs(hprev))
    print "WLSTM", np.sum(np.abs(WLSTM))
    print "xt", np.sum(np.abs(xt))
    print "*****"
    """
    _, cprev, hprev, cache = LSTM.forward(xt, WLSTM, cprev, hprev)
    caches[t] = cache
    Hcat[t] = hprev


  # sanity check: perform batch forward to check that we get the same thing
  print "22222222222222"
  H, _, _, batch_cache = LSTM.forward(X, WLSTM, c0, h0)
  print "H=", np.sum(np.abs(H))
  print "abs diff=", np.sum(np.abs(H - Hcat)), np.sum(np.abs(H))
  assert np.allclose(H, Hcat), 'Sequential and Batch forward don''t match!'

  # eval loss
  #wrand = np.random.randn(*Hcat.shape)
  #print "wrand", wrand.shape
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

  loss = np.sum(Hcat * wrand)
  print "loss, ", loss
  print "wrand", np.sum(np.abs(wrand))
  dH = wrand

  # get the batched version gradients
  BdX, BdWLSTM, Bdc0, Bdh0 = LSTM.backward(dH, batch_cache)
  print "BdX", np.sum(np.abs(BdX))
  print "BdWLSTM", np.sum(np.abs(BdWLSTM))
  print "Bdc0", np.sum(np.abs(Bdc0))
  print "Bdh0", np.sum(np.abs(Bdh0))
  exit(0)

  print "================="
  # now perform sequential backward
  dX = np.zeros_like(X)
  dWLSTM = np.zeros_like(WLSTM)
  dc0 = np.zeros_like(c0)
  dh0 = np.zeros_like(h0)
  dcnext = None
  dhnext = None
  for t in reversed(xrange(n)):
    dht = dH[t].reshape(1, b, d)

    dx, dWLSTMt, dcprev, dhprev = LSTM.backward(dht, caches[t], dcnext, dhnext)
    print "t=", t, "dht=", np.sum(np.abs(dht)), "dcnext=", np.sum(np.abs(dcprev)), "dhnext=", np.sum(np.abs(dhprev))
    dhnext = dhprev
    dcnext = dcprev

    dWLSTM += dWLSTMt # accumulate LSTM gradient
    dX[t] = dx[0]
    if t == 0:
      dc0 = dcprev
      dh0 = dhprev

  # and make sure the gradients match
  print 'Making sure batched version agrees with sequential version: (should all be True)'
  print np.allclose(BdX, dX)
  print np.allclose(BdWLSTM, dWLSTM)
  print np.allclose(Bdc0, dc0)
  print np.allclose(Bdh0, dh0)


def checkBatchGradient():
  """ check that the batch gradient is correct """

  # lets gradient check this beast
  n,b,d = (5, 3, 4) # sequence length, batch size, hidden size
  input_size = 10
  WLSTM = LSTM.init(input_size, d) # input size, hidden size
  X = np.random.randn(n,b,input_size)
  h0 = np.random.randn(b,d)
  c0 = np.random.randn(b,d)

  # batch forward backward
  H, Ct, Ht, cache = LSTM.forward(X, WLSTM, c0, h0)
  wrand = np.random.randn(*H.shape)
  loss = np.sum(H * wrand) # weighted sum is a nice hash to use I think
  dH = wrand
  dX, dWLSTM, dc0, dh0 = LSTM.backward(dH, cache)

  def fwd():
    h,_,_,_ = LSTM.forward(X, WLSTM, c0, h0)
    return np.sum(h * wrand)

  # now gradient check all
  delta = 1e-5
  rel_error_thr_warning = 1e-2
  rel_error_thr_error = 1
  tocheck = [X, WLSTM, c0, h0]
  grads_analytic = [dX, dWLSTM, dc0, dh0]
  names = ['X', 'WLSTM', 'c0', 'h0']
  for j in xrange(len(tocheck)):
    mat = tocheck[j]
    dmat = grads_analytic[j]
    name = names[j]
    # gradcheck
    for i in xrange(mat.size):
      old_val = mat.flat[i]
      mat.flat[i] = old_val + delta
      loss0 = fwd()
      mat.flat[i] = old_val - delta
      loss1 = fwd()
      mat.flat[i] = old_val

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
        if rel_error > rel_error_thr_error: status = '!!!!! NOTOK'

      # print stats
      print '%s checking param %s index %s (val = %+8f), analytic = %+8f, numerical = %+8f, relative error = %+8f' \
            % (status, name, `np.unravel_index(i, mat.shape)`, old_val, grad_analytic, grad_numerical, rel_error)


if __name__ == "__main__":

  checkSequentialMatchesBatch()
  #raw_input('check OK, press key to continue to gradient check')
  #checkBatchGradient()
  #print 'every line should start with OK. Have a nice day!'
