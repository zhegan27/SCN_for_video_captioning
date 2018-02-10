
import numpy as np
import theano
import theano.tensor as tensor
from theano import config

from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import dropout, numpy_floatX
from utils import uniform_weight, zero_bias

# Set the random number generators' seeds for consistency
SEED = 123  
np.random.seed(SEED)

""" init. parameters. """  
def init_params(options):
    
    n_y = options['n_y']
    n_z = options['n_z'] 
    
    params = OrderedDict()
    
    params['Wy1'] = uniform_weight(n_z,512)
    params['by1'] = zero_bias(512) 

    params['Wy2'] = uniform_weight(512,n_y)
    params['by2'] = zero_bias(n_y)                                     

    return params

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams
    
""" Building model... """

def build_model(tparams,options):
    
    trng = RandomStreams(options['SEED'])
    
    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    # size of n_samples * n_z 
    z = tensor.matrix('z', dtype=config.floatX)
    # size of n_samples * n_y 
    y = tensor.matrix('y', dtype=config.floatX)

    z = dropout(z, trng, use_noise)
    
    h = tensor.tanh(tensor.dot(z, tparams['Wy1'])+tparams['by1'])
    h = dropout(h, trng, use_noise)
    
    # size of n_samples * n_y
    pred = tensor.nnet.sigmoid(tensor.dot(h, tparams['Wy2'])+tparams['by2'])
    
    f_pred = theano.function([z],pred,name='f_pred')
                                         
    cost = (-y * tensor.log(pred + 1e-6) - (1.-y) * tensor.log(1. - pred + 1e-6)).sum() / z.shape[0]                            

    return use_noise, z, y, cost, f_pred
