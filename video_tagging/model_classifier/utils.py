import numpy as np
import theano
from theano import config
from collections import OrderedDict

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)
    
def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

     
def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)
    
def _p(pp, name):
    return '%s_%s' % (pp, name)

def dropout(X, trng, p=0.):
    if p != 0:
        retain_prob = 1 - p
        X = X / retain_prob * trng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
    return X

""" used for initialization of the parameters. """

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)
    
def uniform_weight(nin,nout=None, scale=0.05):
    if nout == None:
        nout = nin
    W = np.random.uniform(low=-scale, high=scale, size=(nin, nout))
    return W.astype(config.floatX)
    
def normal_weight(nin,nout=None, scale=0.05):
    if nout == None:
        nout = nin
    W = np.random.randn(nin, nout) * scale
    return W.astype(config.floatX)
    
def zero_bias(ndim):
    b = np.zeros((ndim,))
    return b.astype(config.floatX)







 

    
