# MANGO

### Install dependencies
```
pip install transformers, datasets, text_attack
```

### Downloading GPT-2 trained on BERT tokenizer (optional)
To attack a BERT model, MANGO requires a causal language model trained on the BERT tokenizer, provided in [GBDA official repository](https://github.com/facebookresearch/text-adversarial-attack). Before the attack, please run the following script to download the model from the Amazon S3 bucket:
```
curl https://dl.fbaipublicfiles.com/text-adversarial-attack/transformer_wikitext-103.pth -o transformer_wikitext-103.pth
```

### Perform attack
For instance, to run MANGO on AG News dataset on first 1000 samples, run:
```
python run_attack.py --attack recipes/white-box/mango.py --dataset ag_news --seed 0 --chunk_id 0 --chunk_size 1000
```

### Retrieve results
To retrieve basic results from previous example attack on AG News, run:
```
python retrieve_results.py -d results/ag_news/white-box/mango_0 --device cuda
```

### Original repository

The repository is clone of [GBDA official repository](https://github.com/facebookresearch/text-adversarial-attack). Please cite [[1]](https://arxiv.org/abs/2104.13733) if you found the resources in this repository useful.


[1] C. Guo *, A. Sablayrolles *, Herve Jegou, Douwe Kiela.  [*Gradient-based Adversarial Attacks against Text Transformers*](https://arxiv.org/abs/2104.13733). EMNLP 2021.


```
@article{guo2021gradientbased,
  title={Gradient-based Adversarial Attacks against Text Transformers},
  author={Guo, Chuan and Sablayrolles, Alexandre and Jégou, Hervé and Kiela, Douwe},
  journal={arXiv preprint arXiv:2104.13733},
  year={2021}
}
```

