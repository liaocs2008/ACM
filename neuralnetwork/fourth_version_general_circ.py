#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import time

os.environ["THEANO_FLAGS"] = "device=gpu2"

import numpy as np
import theano
import theano.tensor as T

from theano.tensor import fft

import lasagne

theano.config.exception_verbosity = 'high'
theano.config.optimizer = 'None'



class LeoLayer2(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, W=lasagne.init.Normal(0.01),
                 nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(LeoLayer2, self).__init__(incoming, **kwargs)
        self.nonlinearity = (lasagne.nonlinearities.identity if nonlinearity is None else nonlinearity)

        self.num_inputs = np.product(self.input_shape[1:])
        self.num_units = np.max([num_units, self.num_inputs])
        self.desired_output = num_units
        self.W = self.add_param(W, (1, self.num_units), name='LeoLayer2_W', broadcastable=(True, False))
        print("leo2", self.input_shape, self.num_inputs, self.num_units)

    def get_output_for(self, input, **kwargs):
        fft_w = fft.rfft(T.reshape(self.W, [1, self.num_units]))
        x = fft_w[..., 0]
        y = fft_w[..., 1]

        if self.num_inputs >= self.num_units:
            fft_x = fft.rfft(T.reshape(input, [input.shape[0], self.num_inputs]))
            u = fft_x[..., 0]
            v = fft_x[..., 1]
            u = T.reshape(u, [input.shape[0], self.num_inputs / 2 + 1])
            v = T.reshape(v, [input.shape[0], self.num_inputs / 2 + 1])
        else:
            tmp_x = T.zeros([input.shape[0], self.num_units])
            tmp_x = T.set_subtensor(tmp_x[:, :self.num_inputs], T.reshape(input, [input.shape[0], self.num_inputs]))
            fft_x = fft.rfft(tmp_x)
            u = fft_x[..., 0]
            v = fft_x[..., 1]
            u = T.reshape(u, [input.shape[0], self.num_units / 2 + 1])
            v = T.reshape(v, [input.shape[0], self.num_units / 2 + 1])

        x = T.reshape(x, [self.num_units / 2 + 1])
        y = T.reshape(y, [self.num_units / 2 + 1])
        res = T.stack([u * x - v * y, v * x + u * y], axis=2)
        res = fft.irfft(res)
        res = T.reshape(res, [input.shape[0], self.num_units])
        return self.nonlinearity( res[:,:self.desired_output] )

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.desired_output)



class LeoLayer3(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, W=lasagne.init.Normal(0.01),
                 nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(LeoLayer3, self).__init__(incoming, **kwargs)
        self.nonlinearity = (lasagne.nonlinearities.identity if nonlinearity is None else nonlinearity)

        self.num_inputs = np.product(self.input_shape[1:])
        self.desired_output = num_units
        if num_units < self.num_inputs:
            self.k = int((self.num_inputs + num_units - 1) / num_units)
            self.n = num_units
        else:
            self.k = int((self.num_inputs + num_units - 1) / self.num_inputs)
            self.n = self.num_inputs
        self.num_units = self.k * self.n
        self.W = self.add_param(W, (self.k, 1, self.n), name='LeoLayer3_W', broadcastable=(True, False))
        print("leo efficient", self.k, self.n)

        # a, b = np.ogrid[0:self.num_units, 0:-self.num_units:-1]
        # self.indx = a + b
        # print("leo", self.input_shape, num_units)

    def get_output_for(self, input, **kwargs):
        fft_w = fft.rfft(T.reshape(self.W, [self.k, 1, self.n]))
        x = fft_w[..., 0]
        y = fft_w[..., 1]
        x = T.reshape(x, [self.k, 1, self.n / 2 + 1])
        y = T.reshape(y, [self.k, 1, self.n / 2 + 1])

        if self.desired_output < self.num_inputs:
            tmp_x = T.zeros((input.shape[0], self.num_units), dtype=input.dtype)
            tmp_x = T.set_subtensor(tmp_x[:, :self.num_inputs], T.reshape(input, [input.shape[0], self.num_inputs]))
            tmp_x = T.reshape(tmp_x, [input.shape[0], self.k, self.n])
            tmp_x = T.transpose(tmp_x, [1, 0, 2])
            fft_x = fft.rfft(tmp_x)
            u = fft_x[..., 0]
            v = fft_x[..., 1]
            u = T.reshape(u, [self.k, input.shape[0], self.n / 2 + 1])
            v = T.reshape(v, [self.k, input.shape[0], self.n / 2 + 1])
            r = T.stack([u * x - v * y, v * x + u * y], axis=3)
            r = T.sum(fft.irfft(r), axis=0)
            r = T.reshape(r, [input.shape[0], self.n])
        else:
            tmp_x = T.reshape(input, [input.shape[0], self.n])
            fft_x = fft.rfft(tmp_x)
            u = fft_x[..., 0]
            v = fft_x[..., 1]
            u = T.reshape(u, [input.shape[0], self.n / 2 + 1])
            v = T.reshape(v, [input.shape[0], self.n / 2 + 1])
            r = T.stack([u * x - v * y, v * x + u * y], axis=3)
            r = T.transpose(fft.irfft(r), [1, 0, 2])
            r = T.reshape(r, [input.shape[0], self.num_units])

        return self.nonlinearity( r[:,:self.desired_output] )

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.desired_output)





class LeoLayer4(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, partition_size, W=lasagne.init.Normal(0.01),
                 nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(LeoLayer4, self).__init__(incoming, **kwargs)
        self.nonlinearity = (lasagne.nonlinearities.identity if nonlinearity is None else nonlinearity)

        self.num_inputs = np.product(self.input_shape[1:])
        self.desired_output = num_units
        self.n = partition_size # block size is n x n

        self.rows = (self.desired_output + self.n - 1) / self.n
        self.cols = (self.num_inputs + self.n - 1) / self.n

        self.W = self.add_param(W, (self.rows, self.cols, self.n), name='LeoLayer3_W', broadcastable=(True, False))
        print("leo general circ", self.num_inputs, self.rows, self.cols, self.n)


    def get_output_for(self, input, **kwargs):
        fft_w = fft.rfft(T.reshape(self.W, [self.rows * self.cols, self.n]))
        x = fft_w[..., 0]
        y = fft_w[..., 1]
        x = T.reshape(x, [self.rows, self.cols, self.n / 2 + 1])
        y = T.reshape(y, [self.rows, self.cols, self.n / 2 + 1])

        tmp_x = T.zeros((input.shape[0], self.cols * self.n), dtype=input.dtype)
        tmp_x = T.set_subtensor(tmp_x[:, :self.num_inputs], T.reshape(input, [input.shape[0], self.num_inputs]))
        tmp_x = T.reshape(tmp_x, [input.shape[0] * self.cols, self.n])
        fft_x = fft.rfft(tmp_x)
        u = fft_x[..., 0]
        v = fft_x[..., 1]
        u = T.reshape(u, [input.shape[0], 1, self.cols, self.n / 2 + 1])
        v = T.reshape(v, [input.shape[0], 1, self.cols, self.n / 2 + 1])
        r = T.stack([u * x - v * y, v * x + u * y], axis=4)
        r = T.reshape(r, [input.shape[0] * self.rows * self.cols, self.n / 2 + 1, 2])
        r = fft.irfft(r)
        r = T.reshape(r, [input.shape[0], self.rows, self.cols, self.n])
        r = T.sum(r, axis=2)
        r = T.reshape(r, [input.shape[0], self.rows * self.n])
        return self.nonlinearity( r[:,:self.desired_output] )

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.desired_output)







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
            download(filename)
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



def build_newcirc_cnn(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = LeoLayer4(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            partition_size=8,
            nonlinearity=lasagne.nonlinearities.rectify, name="leolayer4_1")

    network = LeoLayer4(
        lasagne.layers.dropout(network, p=.5),
        partition_size=2,
        num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax, name="leolayer4_2")

    return network



def build_dense_cnn(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network




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


def main(model='cnn_newcirc', num_epochs=10, model_name_prefix='0'):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model == 'cnn_newcirc':
        network = build_newcirc_cnn(input_var)
    elif model == 'cnn_dense':
        network = build_dense_cnn(input_var)
    else:
        print("Unrecognized model type %r." % model)
        return

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)

    #theano.printing.pydotprint(prediction, outfile="leolayer2.png", var_with_name_simple=True)

    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

    #theano.printing.pydotprint(updates, outfile="leolayer2_2.png", var_with_name_simple=True)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)


    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    # print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            #train_err += train_fn(inputs, targets)
            e = train_fn(inputs, targets)
            train_err += e
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results for model %s:" % model_name_prefix)
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)
    np.savez(model_name_prefix + '.npz', *lasagne.layers.get_all_param_values(network))


def generate_model(num_committee=5):
    for i in xrange(num_committee):
        main('cnn', 10, '%s' % i)


def get_models(num_committee=5):
    models = []
    for i in xrange(num_committee):
        with np.load('%d.npz' % i) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            models.append(param_values)
    return models



if __name__ == '__main__':
    main('cnn_newcirc', 100, 'mnist_newcirc_replace_largest')
    #main('cnn_dense', 100, 'mnist_dense_replace_largest')



