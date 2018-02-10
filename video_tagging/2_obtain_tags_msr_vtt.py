import numpy as np
import cPickle

if __name__ == "__main__":
    
    gt_tag = cPickle.load(open("./data/gt_tag_feats_youtube2text.p",'rb'))
    wordtoix_tag, ixtoword_tag = gt_tag[3], gt_tag[4]
    del gt_tag
    
    path = '/home2/Data/video_caption/MSR_VTT/'
    
    x = cPickle.load(open("./data/corpus_msr_vtt.p","rb"))
    train, valid, test= x['train'], x['valid'], x['test']
    wordtoix_msrvtt, ixtoword_msrvtt = x['wordtoix'], x['ixtoword']
    del x
    
    
    MSR_VTT_tag_feats = np.zeros((7010,len(ixtoword_tag)))
    
    for i in range(len(train[0])):
        for word in train[0][i]:
            ix = train[1][i]
            if ixtoword_msrvtt[word] in wordtoix_tag:
                iy = wordtoix_tag[ixtoword_msrvtt[word]]
                MSR_VTT_tag_feats[ix, iy] = 1
                
                
    for i in range(len(test[0])):
        for word in test[0][i]:
            ix = test[1][i]
            if ixtoword_msrvtt[word] in wordtoix_tag:
                iy = wordtoix_tag[ixtoword_msrvtt[word]]
                MSR_VTT_tag_feats[ix, iy] = 1
                
                
                    
    for i in range(len(valid[0])):
        for word in valid[0][i]:
            ix = valid[1][i]
            if ixtoword_msrvtt[word] in wordtoix_tag:
                iy = wordtoix_tag[ixtoword_msrvtt[word]]
                MSR_VTT_tag_feats[ix, iy] = 1
    
    cPickle.dump(MSR_VTT_tag_feats, open("./data/gt_tag_feats_msr_vtt.p", "wb"))
    
                
    
    
    
        

    
