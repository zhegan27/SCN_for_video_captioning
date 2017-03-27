# SCN for video captioning

This repo contains the code of using SCN for video captioning, based on the CVPR 2017 paper “[Semantic Compositional Networks for Visual Captioning](https://arxiv.org/pdf/1611.08002.pdf)”. 

To keep things simple, SCN for image captioning is provided in [another separate repo](https://github.com/zhegan27/Semantic_Compositional_Nets).

## Dependencies

This code is written in python. To use it you will need:

* Python 2.7 (do not use Python 3.0)
* Theano 0.7 (you can also use the most recent version)
* A recent version of NumPy and SciPy 

## Getting started

We provide the code on how to train SCN for video captioning on the Youtube2Text dataset. The SCN used in this experiment is a slightly different version of the original SCN. That is, we feed the video features to each step of the SCN-LSTM, instead of only the first step.  

* In order to start, please first download the [C3D, ResNet features and tag features](https://drive.google.com/open?id=0B1HR6m3IZSO_amVjYXFyaXctSjA) for the Youtube2Text dataset we used in the experiments. Put the  `youtube2text` folder inside the `./data` folder.

* We also provide our [pre-trained model](https://drive.google.com/open?id=0B1HR6m3IZSO_amVjYXFyaXctSjA) on Youtube2Text. Put the `pretrained_model` folder into the current directory.

* In order to evaluate the model, please download the standard [coco-caption evaluation code](https://github.com/tylin/coco-caption). Copy the folder `pycocoevalcap` into the current directory.

* Now, everything is ready.

## How to use the code

1. Run `SCN_training.py` to start training.
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python SCN_training.py 
```

2. Based on our pre-trained model, run `SCN_decode.py` to generate captions on the Youtube2Text test set. The generated captions are also provided, named `youtube2text_scn_test.txt`.

3. Now, run `SCN_evaluation.py` to evaluate the model. The code will output

```
CIDEr: 0.777, Bleu-4: 0.511, Bleu-3: 0.606, Bleu-2: 0.697, Bleu-1: 0.810, ROUGE_L: 0.706, METEOR: 0.335. 
```

## Citing SCN

Please cite our CVPR paper in your publications if it helps your research:

    @inproceedings{SCN_CVPR2017,
      Author = {Gan, Zhe and Gan, Chuang and He, Xiaodong and Pu, Yunchen and Tran, Kenneth and Gao, Jianfeng and Carin, Lawrence and Deng, Li},
      Title = {Semantic Compositional Networks for Visual Captioning},
      booktitle={CVPR},
      Year  = {2017}
    }




