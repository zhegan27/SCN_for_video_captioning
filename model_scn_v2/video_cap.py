
#import numpy as np
import theano
import theano.tensor as tensor
from theano import config

from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import dropout, numpy_floatX
from utils import uniform_weight, zero_bias

from lstm_layers import param_init_encoder, encoder_layer

# Set the random number generators' seeds for consistency
#SEED = 123  
#np.random.seed(SEED)

""" init. parameters. """  
def init_params(options,W):
    
    n_words = options['n_words']
    n_x = options['n_x']  
    n_h = options['n_h']
    n_z = options['n_z'] 
    
    params = OrderedDict()
    # word embedding
    # params['Wemb'] = uniform_weight(n_words,n_x)
    params['Wemb'] = W.astype(config.floatX)
    params = param_init_encoder(options,params)
    
    params['Vhid'] = uniform_weight(n_h,n_x)
    params['bhid'] = zero_bias(n_words) 
    
    params['C0'] = uniform_weight(n_z, n_x)
                                       
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

    # input sentences, size of n_steps * n_samples
    x = tensor.matrix('x', dtype='int64')
    # the corresponding masks padding zeros
    mask = tensor.matrix('mask', dtype=config.floatX)
    # size of n_samples * n_z
    z = tensor.matrix('z', dtype=config.floatX)
    y = tensor.matrix('y', dtype=config.floatX)
    z = dropout(z, trng, use_noise)
    y = dropout(y, trng, use_noise)

    n_steps = x.shape[0] # the sentence length in this mini-batch
    n_samples = x.shape[1] # the number of sentences in this mini-batch
    
    n_x = tparams['Wemb'].shape[1] # the dimension of the word embedding
    
    # size of n_steps,n_samples,n_x
    emb = tparams['Wemb'][x.flatten()].reshape([n_steps,n_samples,n_x])
    emb = dropout(emb, trng, use_noise)
    
    # 1 * n_samples * n_x
    z0 =tensor.dot(z,tparams['C0']).dimshuffle('x',0,1)
    # n_steps * n_samples * n_x
    emb_input = tensor.concatenate((z0,emb[:n_steps-1]))
    # n_steps * n_samples
    mask0 =mask[0].dimshuffle('x',0)
    mask_input = tensor.concatenate((mask0,mask[:n_steps-1]))

    # decoding the sentence vector z back into the original sentence
    h_decoder = encoder_layer(tparams, emb_input, mask_input,y,z,seq_output=True)
    h_decoder = dropout(h_decoder, trng, use_noise)
                                         
    shape = h_decoder.shape
    h_decoder = h_decoder.reshape((shape[0]*shape[1], shape[2]))
    
    Vhid = tensor.dot(tparams['Vhid'],tparams['Wemb'].T)
    pred_x = tensor.dot(h_decoder, Vhid) + tparams['bhid']
    pred = tensor.nnet.softmax(pred_x)
    
    x_vec = x.reshape((shape[0]*shape[1],))
    
    index = tensor.arange(shape[0]*shape[1])
    
    pred_word = pred[index, x_vec]
    mask_word = mask.reshape((shape[0]*shape[1],))
    
    index_list = theano.tensor.eq(mask_word, 1.).nonzero()[0]
    
    pred_word = pred_word[index_list]
    
    # the cross-entropy loss                 
    cost = -tensor.log(pred_word + 1e-6).sum() / n_samples  
    
    return use_noise, x, mask, y, z, cost
