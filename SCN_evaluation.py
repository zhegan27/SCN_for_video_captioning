"""
Semantic Compositional Network https://arxiv.org/pdf/1611.08002v1.pdf
Developed by Zhe Gan, zg27@duke.edu, Sep., 2016

Computes the BLEU, ROUGE, METEOR, and CIDER
using the COCO metrics scripts
"""

# this requires the coco-caption package, https://github.com/tylin/coco-caption
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

import cPickle

def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

if __name__ == '__main__':
    
    # this is the generated captions
    hypo = {idx: [lines.strip()] for (idx, lines) in enumerate(open('./youtube2text_scn_test.txt', 'rb') )}  
    
    # this is the ground truth captions
    x = cPickle.load(open("./data/youtube2text/references.p","rb"))
    refs = x[2]
    del x
    
    refs = {idx: ref for (idx, ref) in enumerate(refs)}
    
    print score(refs, hypo)
    
    
    
    
        

