# Contrastive Knowledge Graph Error Detection
Pytorch-based implementation of CAGED, and the description of the model and the results can be found in the CIKM 2022 paper: "Contrastive Knowledge Graph Error Detection"[1].

## Requirements

All the required packages can be installed by running `pip install -r requirements.txt`.

## Inductive Knowledge Graph Completion

To replicate the experiments from our paper[1]:

Train：

`python Our_TopK%_RankingList.py “train”`


Test：

`python Our_TopK%_RankingList.py “test”`


Case study:

`python casestudy.py variant=var_n`


Note that the default values of hyperparameters were listed in `param.py`, while the models are in `model.py`.
