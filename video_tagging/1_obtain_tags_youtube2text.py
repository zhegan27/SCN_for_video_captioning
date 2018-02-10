import numpy as np
import cPickle

if __name__ == "__main__":
    
    x = cPickle.load(open("./data/corpus_youtube2text.p","rb"))
    train, val, test = x[0], x[1], x[2]
    wordtoix, ixtoword = x[3], x[4]
    del x
    n_words = len(ixtoword)
    
    n_count = np.zeros((n_words,)).astype("int32")
    for sent in train[0]:
        for w in sent:
            n_count[w] = n_count[w] + 1
    
    for sent in val[0]:
        for w in sent:
            n_count[w] = n_count[w] + 1
    
    idx = np.argsort(n_count)[::-1]
    
    count_sorted = np.sort(n_count)[::-1]
    word_sorted = []
    for i in idx:
        word_sorted.append(ixtoword[i])
    
    # manually select tags that you think are important and useful 
    # here, we manually select 300 tags when we did the experiments
    selected = [4,5,8,13,16,17,20,21,23,24,26,27,28,29,30,31,32,34,35,36,38,41,42] + \
                range(44,51) + range(52,56) + [57,59,62,63,66,70,71,72,74] + \
                [76,77,78,80,82,83,84,85,87,88,90,91,93,94,95,96,97,98,99,100,101] + \
                [102,103,106,108,109,110,111,112,113,114,115,116,117,120,121,122,123] + \
                [124,125,126,127,128,129,130,131,133,136,138,139,140,141,142] + \
                [143,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160] +\
                range(162,173) + [175,176,177,178,179,180,181,183,185,186,188,189] +\
                range(190,203) + range(204,215) + range(216,224) + [226,227,228,230] +\
                [231,232,233,234,236] + range(238,253) + range(253,258) + [259,260,261] +\
                [263,265] + range(268,278) + [279,282,283,284,286,288,289,290] +\
                range(291,309) + range(310,313) + [315,317,318,319,320,321,322] + [326,327] +\
                [328,329,331,332,334,335,336,338,339,340] + range(341,363) +[364,366,367] +\
                range(368,374) + [376,377,378,379,380,381,382,383,384]
                
    #print len(selected)
    #print selected
    
    key_words = []
    for i in selected:
        key_words.append(word_sorted[i])
    
    ixtoword = {}
    wordtoix = {}
    
    for idx in range(len(key_words)):
        wordtoix[key_words[idx]] = idx
        ixtoword[idx] = key_words[idx]
        
    x = cPickle.load(open("./data/references_youtube2text.p","rb"))
    train_refs, valid_refs, test_refs = x[0], x[1], x[2]
    del x
    
    train_label = np.zeros((len(train_refs),300))
    for i in range(len(train_refs)):
        sents = train_refs[i]
        for sent in sents:
            words = sent.split(" ")
            for w in words:
                if w in wordtoix:
                    train_label[i,wordtoix[w]] = 1.
                    
    valid_label = np.zeros((len(valid_refs),300))
    for i in range(len(valid_refs)):
        sents = valid_refs[i]
        for sent in sents:
            words = sent.split(" ")
            for w in words:
                if w in wordtoix:
                    valid_label[i,wordtoix[w]] = 1.
                    
    test_label = np.zeros((len(test_refs),300))
    for i in range(len(test_refs)):
        sents = test_refs[i]
        for sent in sents:
            words = sent.split(" ")
            for w in words:
                if w in wordtoix:
                    test_label[i,wordtoix[w]] = 1.
                    
    cPickle.dump([train_label, valid_label, test_label, wordtoix, ixtoword], open("./data/gt_tag_feats_youtube2text.p", "wb"))
        
    
    
    
    
    
    
    
        

    
