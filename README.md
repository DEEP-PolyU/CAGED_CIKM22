# Contrastive Knowledge Graph Error Detection
Pytorch-based implementation of CAGED, and the description of the model and the results can be found in the paper: "Contrastive Knowledge Graph Error Detection".

## Requirements

All the required packages can be installed by running `pip install -r requirements.txt`.

## Knowledge Graph Error Detection

Data pre-processing:

Following the previous study[1,2], we employ three real-world datasets that are constructed with noisy triples to be 5%, 10% and 15% of the whole KGs based on the popular benchmarks, i.e. FB15k, WN18RR and NELL-995.

To replicate the experiments from our paper:

Train：

`python Our_TopK%_RankingList.py --dataset "WN18RR" --mode “train” --anomaly_ratio 0.05 --mu 0.001 --lam 0.1`


Test：

`python Our_TopK%_RankingList.py --dataset "WN18RR" --mode “test” --anomaly_ratio 0.05 --mu 0.001 --lam 0.1`



## Acknowledgment
This repo is built upon the following work:
```
[1] "Triple trustworthiness measurement for knowledge graph".
https://github.com/TJUNLP/TTMF.git

[2] "Does william shakespeare REALLY write hamlet? knowledge representation learning with confidence".
https://github.com/GemsLab/KGist.git
```


