"""
Semantic Compositional Network https://arxiv.org/pdf/1611.08002.pdf
Developed by Zhe Gan, zg27@duke.edu, Sep., 2016
"""

import time
import logging
import cPickle

import numpy as np
import scipy.io
import theano
import theano.tensor as tensor

from model_classifier.classifier import init_params, init_tparams, build_model
from model_classifier.optimizers import Adam
from model_classifier.utils import get_minibatches_idx, zipp, unzip

# Set the random number generators' seeds for consistency
SEED = 123  
np.random.seed(SEED)

def calu_negll(f_cost, data_x, data_y, iterator):

    totalcost = 0.
    totallen = 0.
    for _, valid_index in iterator:
        z = data_x[valid_index]
        y = data_y[valid_index]
                
        cost = f_cost(z,y) * z.shape[0]
        totalcost += cost
        totallen +=  z.shape[0]
    return totalcost/totallen


""" Training the model. """
if __name__ == '__main__':
    
    logger = logging.getLogger('training_video_tagging_log_file')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('training_video_tagging_log_file.log')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    
    data = scipy.io.loadmat('./data/resnet_feats_youtube2text.mat')
    img_feats = data['feature'].T.astype(theano.config.floatX)
    
    x = cPickle.load(open("./data/gt_tag_feats_youtube2text.p","rb"))
    train_tag = x[0].astype(theano.config.floatX)
    valid_tag = x[1].astype(theano.config.floatX)
    test_tag = x[2].astype(theano.config.floatX)
    del x
        
    ''' MSR-VTT '''
    train_y_more = cPickle.load(open("./data/gt_tag_feats_msr_vtt.p","rb"))
    train_y_more = train_y_more.astype(theano.config.floatX)
    
    x = scipy.io.loadmat('./data/train_feats_msr_vtt.mat') 
    tr_feats = x['feats']
    x = scipy.io.loadmat('./data/valid_feats_msr_vtt.mat') 
    val_feats = x['feats']
    msr_train_resnet_img_feats = np.concatenate((tr_feats, val_feats),axis=0).astype(theano.config.floatX)
    train_x_more = msr_train_resnet_img_feats
    
    max_epochs=100
    lrate=0.0002
    batch_size=128
    valid_batch_size=128
    dropout_val=0.5
    dispFreq=10
    validFreq=10
    saveFreq=100
    patient = 100
    
    saveto = 'youtube_tagging_learned_params.npz'

    options = {}
    options['max_epochs'] = max_epochs
    options['lrate'] = lrate
    options['batch_size'] = batch_size
    options['valid_batch_size'] = valid_batch_size
    options['dispFreq'] = dispFreq
    options['validFreq'] = validFreq
    options['saveFreq'] = saveFreq
    
    options['n_z'] = img_feats.shape[1]
    options['n_y'] = train_tag.shape[1]
    options['SEED'] = SEED
    
    train_x = img_feats[:1200]
    valid_x = img_feats[1200:1300]
    test_x = img_feats[1300:]
    
    train_y = train_tag
    valid_y = valid_tag
    test_y = test_tag
   

    logger.info('Model options {}'.format(options))
    logger.info('{} train examples'.format(train_y.shape[0]))
    logger.info('{} valid examples'.format(valid_y.shape[0]))
    logger.info('{} test examples'.format(test_y.shape[0]))

    logger.info('Building model...')
    
    params = init_params(options)
    tparams = init_tparams(params)

    (use_noise, z, y, cost, f_pred) = build_model(tparams,options)
    
    f_cost = theano.function([z,y], cost, name='f_cost')
    
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = Adam(tparams, cost, [z,y], lr)

    logger.info('Training model...')

    kf_valid = get_minibatches_idx(valid_y.shape[0], valid_batch_size)
    kf_test = get_minibatches_idx(test_y.shape[0], valid_batch_size)
    
    estop = False  # early stop
    history_negll = []
    best_p = None
    bad_counter = 0    
    uidx = 0  # the number of update done
    start_time = time.time()
    kf_1 = get_minibatches_idx(train_y.shape[0], batch_size, shuffle=True)
    kf_2 = get_minibatches_idx(train_y_more.shape[0], batch_size, shuffle=True)
    length_main = len(kf_1)
    length_more = len(kf_2)
    length_iter = max(length_main, length_more)
    try:
        for eidx in xrange(max_epochs):
            kf_1 = get_minibatches_idx(train_y.shape[0], batch_size, shuffle=True)
            kf_2 = get_minibatches_idx(train_y_more.shape[0], batch_size, shuffle=True)
            for iter_num in range(length_iter):
                i1 = np.mod(iter_num, length_main)
                train_index_main = kf_1[i1][1]
                i2 = np.mod(iter_num, length_main)
                train_index_more = kf_2[i2][1]
                
                uidx += 1
                use_noise.set_value(dropout_val)
                
                z = np.concatenate((train_x[train_index_main], train_x_more[train_index_more]),axis=0)
                y = np.concatenate((train_y[train_index_main], train_y_more[train_index_more]),axis=0)

                cost = f_grad_shared(z,y)
                f_update(lrate)

                if np.isnan(cost) or np.isinf(cost):
                    logger.info('NaN detected')
                    break

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
                    
                    valid_negll = calu_negll(f_cost,  valid_x, valid_y,  kf_valid)
                    test_negll = calu_negll(f_cost,  test_x, test_y,  kf_test)
                    history_negll.append([valid_negll, test_negll])
                    
                    if (uidx == 0 or
                        valid_negll <= np.array(history_negll)[:,0].min()):
                             
                        best_p = unzip(tparams)
                        bad_counter = 0
                        
                    logger.info('Cross Entropy: Valid {} Test {}'.format(valid_negll, test_negll))

                    if (len(history_negll) > patient and
                        valid_negll >= np.array(history_negll)[:-10,0].min()):
                            bad_counter += 1
                            if bad_counter > patient:
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
    valid_negll = calu_negll(f_cost, valid_x, valid_y,  kf_valid)
    test_negll = calu_negll(f_cost, test_x, test_y,  kf_test)
    
    tag_feats = f_pred(img_feats)
    scipy.io.savemat("./tag_feats_pred.mat",{'feats':tag_feats})
    logger.info('Final Results...')
    logger.info('Cross Entropy:Valid {} Test {}'.format(valid_negll, test_negll))
    np.savez(saveto, history_negll=history_negll, **best_p)

    
    logger.info('The code run for {} epochs, with {} sec/epochs'.format(eidx + 1, 
                 (end_time - start_time) / (1. * (eidx + 1))))
    


    
