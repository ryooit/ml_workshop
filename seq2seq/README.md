# Seq2seq
Pytorch implementation of Sutskever et al. (2014) paper (https://arxiv.org/pdf/1409.3215.pdf)

## Requirements
- Python 3
- torch >= 1.0
- torchtext
- torchnlp
- spacy

## Contents

To run the seq2seq model, run 
```
 python run.py --batch_size 128 --emb_dim 32 --hid_dim 32 --n_layers 1 --dropout 0.5 --epochs 10 --lr 0.001
```

Got BLEU score 23.95 with the following script.
```
python run.py --batch_size 128 --emb_dim 256 --hid_dim 256 --n_layers 2 --dropout 0.5 --epochs 50 --lr 0.001
```
