import numpy as np
import cPickle
    
def get_W(w2v, word2idx, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(w2v)
    W = np.zeros(shape=(vocab_size, k))            
    
    for word in w2v:
        W[word2idx[word]] = w2v[word]
    
    return W

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  


if __name__=="__main__":    
    w2v_file = 'GoogleNews-vectors-negative300.bin'
    
    x = cPickle.load(open("./data/youtube2text/corpus.p","rb"))
    #train, val, test = x[0], x[1], x[2]
    wordtoix, ixtoword = x[3], x[4]
    del x
    n_words = len(ixtoword)
      
    w2v = load_bin_vec(w2v_file, wordtoix)
    add_unknown_words(w2v, wordtoix)
    W = get_W(w2v,wordtoix)
    
    #rand_vecs = {}
    #add_unknown_words(rand_vecs, wordtoix)
    #W2 = get_W(rand_vecs,wordtoix)
    
    cPickle.dump([W,wordtoix], open("word2vec.p", "wb"))
    print "pretrained word vector created!"
    
