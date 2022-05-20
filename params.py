import argparse
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


data_dir_FB = "./data/FB15k-237"
data_dir_WN = "./data/WN18RR"
data_dir_umls = "./data/umls"
data_dir_kinship = "./data/kinship"
data_dir_NELL_Triple = "./data/NELL-995"
data_dir_NELL = "./data/NELL"
data_dir_YAGO = "./data/YAGO3-10"
data_dir_FB5M = "./data/FB5M"
data_dir_FB2M = "./data/FB2M"
data_dir_DBPEDIA = "./data/DBPEDIA"


dir_emb_ent = "entity2vec.txt"
dir_emb_rel = "relation2vec.txt"

out_folder = "./checkpoints"
log_folder = "./log"

BiLSTM_input_size = 100
BiLSTM_hidden_size = 100
BiLSTM_num_layers = 2
BiLSTM_num_classes = 1
num_train = 272115
batch_size = 1024
num_neighbor = 39

# BiLSTM
# batch_size_lstm = 160
input_size_lstm = 100
hidden_size_lstm = 100
num_layers_lstm = 4
num_class_lstm = 2
start_id_lstm = 0
num_epochs = 2740

# BiLSTM_Attention
alpha = 0.2
dropout = 0.6


learning_rate = 0.003
gama = 1.0
lam = 0.1
lam1 = 0.1
lam2 = 0.1
anomaly_ratio = 0.01




# Translation_model
total_ent = 0
total_rel = 0
embedding_dim = 100
margin = 1.0
p_norm = 2
lr = 0.01

num_epochs_trans = 10


kkkkk = 1

num_anomaly_num = 300