# Contrastive Knowledge Graph Error Detection
Pytorch-based implementation of CAGED, and the description of the model and the results can be found in the paper: "Contrastive Knowledge Graph Error Detection"[1].

## Requirements

All the required packages can be installed by running `pip install -r requirements.txt`.

## Knowledge Graph Error Detection

Data pre-processing:

Following the previous study[2,3], we employ three real-world datasets that are constructed with noisy triples to be 5%, 10% and 15% of the whole KGs based on the popular benchmarks, i.e. FB15k, WN18RR and NELL-995.

To replicate the experiments from our paper[1]:

Train：

`python Our_TopK%_RankingList.py “train”`


Test：

`python Our_TopK%_RankingList.py “test”`



## Acknowledgment
This repo is built upon the following work:
```
[2] Triple trustworthiness measurement for knowledge graph
https://github.com/TJUNLP/TTMF

[3] "Does william shakespeare REALLY write hamlet? knowledge representation learning with confidence".
https://github.com/GemsLab/KGist.git
```


