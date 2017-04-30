'''
Semantic Compositional Network https://arxiv.org/pdf/1611.08002.pdf
Developed by Zhe Gan, zg27@duke.edu, Sep., 2016
'''

import time
import logging
import cPickle

import numpy as np
import scipy.io
import theano
import theano.tensor as tensor

from model_scn_v2.video_cap import init_params, init_tparams, build_model
from model_scn_v2.optimizers import Adam
from model_scn_v2.utils import get_minibatches_idx, zipp, unzip

# Set the random number generators' seeds for consistency
SEED = 123  
np.random.seed(SEED)

def prepare_data(seqs):
    
    # x: a list of sentences
    lengths = [len(s) for s in seqs]
    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples)).astype('int64')
    x_mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask

def calu_negll(f_cost, prepare_data, data, img_feats, tag_feats, iterator):

    totalcost = 0.
    totallen = 0.
    for _, valid_index in iterator:
        x = [data[0][t]for t in valid_index]
        x, mask = prepare_data(x)
        y = np.array([tag_feats[data[1][t]]for t in valid_index])
        z = np.array([img_feats[data[1][t]]for t in valid_index])
                
        length = np.sum(mask)
        cost = f_cost(x, mask,y,z) * x.shape[1]
        totalcost += cost
        totallen += length
    return totalcost/totallen


""" Training the model. """

def train_model(train, valid, test, img_feats, tag_feats, W, n_words=7164, n_x=300, n_h=512,
    n_f=512, max_epochs=20, lrate=0.0002, batch_size=64, valid_batch_size=64, 
    dropout_val=0.5, dispFreq=10, validFreq=200, saveFreq=1000, 
    saveto = 'youtube_result_scn.npz'):
        
    """ n_words : vocabulary size
        n_x : word embedding dimension
        n_h : LSTM/GRU number of hidden units 
        n_f : number of factors used in SCN
        max_epochs : The maximum number of epoch to run
        lrate : learning rate
        batch_size : batch size during training
        valid_batch_size : The batch size used for validation/test set
        dropout_val : the probability of dropout
        dispFreq : Display to stdout the training progress every N updates
        validFreq : Compute the validation error after this number of update.
        saveFreq : save results after this number of update.
        saveto : where to save.
    """

    options = {}
    options['n_words'] = n_words
    options['n_x'] = n_x
    options['n_h'] = n_h
    options['n_f'] = n_f
    options['max_epochs'] = max_epochs
    options['lrate'] = lrate
    options['batch_size'] = batch_size
    options['valid_batch_size'] = valid_batch_size
    options['dispFreq'] = dispFreq
    options['validFreq'] = validFreq
    options['saveFreq'] = saveFreq
    
    options['n_z'] = img_feats.shape[1]
    options['n_y'] = tag_feats.shape[1]
    options['SEED'] = SEED
   
    logger.info('Model options {}'.format(options))
    logger.info('{} train examples'.format(len(train[0])))
    logger.info('{} valid examples'.format(len(valid[0])))
    logger.info('{} test examples'.format(len(test[0])))

    logger.info('Building model...')
    
    params = init_params(options,W)
    tparams = init_tparams(params)

    (use_noise, x, mask, y, z, cost) = build_model(tparams,options)
    
    f_cost = theano.function([x, mask, y, z], cost, name='f_cost')
    
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = Adam(tparams, cost, [x, mask, y, z], lr)

    logger.info('Training model...')

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)
    
    estop = False  # early stop
    history_negll = []
    best_p = None
    bad_counter = 0    
    uidx = 0  # the number of update done
    start_time = time.time()
    
    try:
        for eidx in xrange(max_epochs):
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(dropout_val)

                x = [train[0][t]for t in train_index]
                y = np.array([tag_feats[train[1][t]]for t in train_index])
                z = np.array([img_feats[train[1][t]]for t in train_index])
                
                x, mask = prepare_data(x)

                cost = f_grad_shared(x, mask,y,z)
                f_update(lrate)

                if np.isnan(cost) or np.isinf(cost):
                    logger.info('NaN detected')
                    return 1., 1., 1.

                if np.mod(uidx, dispFreq) == 0:
                    logger.info('Epoch {} Update {} Cost {}'.format(eidx, uidx, cost))
                    
                if np.mod(uidx, saveFreq) == 0:
                    logger.info('Saving ...')
                
                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    np.savez(saveto, history_negll=history_negll, **params)
                    logger.info('Done ...')

                if np.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    
                    #train_negll = calu_negll(f_cost, prepare_data, train, img_feats, kf)
                    valid_negll = calu_negll(f_cost, prepare_data, valid, img_feats, tag_feats, kf_valid)
                    test_negll = calu_negll(f_cost, prepare_data, test, img_feats, tag_feats, kf_test)
                    history_negll.append([valid_negll, test_negll])
                    
                    if (uidx == 0 or
                        valid_negll <= np.array(history_negll)[:,0].min()):
                             
                        best_p = unzip(tparams)
                        bad_counter = 0
                        
                    logger.info('Perp: Valid {} Test {}'.format(np.exp(valid_negll), np.exp(test_negll)))

                    if (len(history_negll) > 10 and
                        valid_negll >= np.array(history_negll)[:-10,0].min()):
                            bad_counter += 1
                            if bad_counter > 10:
                                logger.info('Early Stop!')
                                estop = True
                                break

            if estop:
                break

    except KeyboardInterrupt:
        logger.info('Training interupted')

    end_time = time.time()
    
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)
        
    use_noise.set_value(0.)
    #kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    #train_negll = calu_negll(f_cost, prepare_data, train, img_feats, kf_train_sorted)
    valid_negll = calu_negll(f_cost, prepare_data, valid, img_feats, tag_feats, kf_valid)
    test_negll = calu_negll(f_cost, prepare_data, test, img_feats, tag_feats, kf_test)
    
    logger.info('Final Results...')
    logger.info('Perp: Valid {} Test {}'.format(np.exp(valid_negll), np.exp(test_negll)))
    np.savez(saveto, history_negll=history_negll, **best_p)

    
    logger.info('The code run for {} epochs, with {} sec/epochs'.format(eidx + 1, 
                 (end_time - start_time) / (1. * (eidx + 1))))
    
    return valid_negll, test_negll

if __name__ == '__main__':
    
    # https://docs.python.org/2/howto/logging-cookbook.html
    logger = logging.getLogger('eval_youtube_scn')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('eval_youtube_scn.log')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    
    x = cPickle.load(open("./data/youtube2text/corpus.p","rb"))
    train, val, test = x[0], x[1], x[2]
    wordtoix, ixtoword = x[3], x[4]
    W = x[5]
    del x
    n_words = len(ixtoword)
    
    data = scipy.io.loadmat('./data/youtube2text/c3d_feats.mat')
    c3d_img_feats = data['feats'].astype(theano.config.floatX)
    
    data = scipy.io.loadmat('./data/youtube2text/resnet_feats.mat')
    resnet_img_feats = data['feature'].T.astype(theano.config.floatX)
    
    img_feats = np.concatenate((c3d_img_feats,resnet_img_feats),axis=1)
    
    data = scipy.io.loadmat('./tag_feats.mat')
    tag_feats = data['feats'].astype(theano.config.floatX)

    [val_negll, te_negll] = train_model(train, val, test, img_feats, tag_feats, W,
        n_words=n_words)
        
