import numpy as np
# import args
from dataset import Reader
# import utils
from create_batch import get_pair_batch_train, get_pair_batch_test, toarray, get_pair_batch_train_common, toarray_float
import torch
from model import BiLSTM_Attention
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
import os
import logging
import math
from matplotlib import pyplot as plt
# import time
import argparse
import random

def main():
    parser = argparse.ArgumentParser(add_help=False)
    # args, _ = parser.parse_known_args()
    parser.add_argument('--model', default='CAGED', help='model name')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], help='run training or evaluation')
    parser.add_argument('-ds', '--dataset', default='WN18RR', help='dataset')
    args, _ = parser.parse_known_args()
    parser.add_argument('--save_dir', default=f'./checkpoints/{args.dataset}/', help='model output directory')
    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.add_argument('--load_model_path', default=f'./checkpoints/{args.dataset}')
    parser.add_argument('--log_folder', default=f'./checkpoints/{args.dataset}/', help='model output directory')


    # data
    parser.add_argument('--data_path', default=f'./data/{args.dataset}/', help='path to the dataset')
    parser.add_argument('--dir_emb_ent', default="entity2vec.txt", help='pretrain entity embeddings')
    parser.add_argument('--dir_emb_rel', default="relation2vec.txt", help='pretrain entity embeddings')
    parser.add_argument('--num_batch', default=2740, type=int, help='number of batch')
    parser.add_argument('--num_train', default=0, type=int, help='number of triples')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--total_ent', default=0, type=int, help='number of entities')
    parser.add_argument('--total_rel', default=0, type=int, help='number of relations')

    # model architecture
    parser.add_argument('--BiLSTM_input_size', default=100, type=int, help='BiLSTM input size')
    parser.add_argument('--BiLSTM_hidden_size', default=100, type=int, help='BiLSTM hidden size')
    parser.add_argument('--BiLSTM_num_layers', default=2, type=int, help='BiLSTM layers')
    parser.add_argument('--BiLSTM_num_classes', default=1, type=int, help='BiLSTM class')
    parser.add_argument('--num_neighbor', default=39, type=int, help='number of neighbors')
    parser.add_argument('--embedding_dim', default=100, type=int, help='embedding dim')

    # regularization
    parser.add_argument('--alpha', type=float, default=0.2, help='hyperparameter alpha')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout for EaGNN')

    # optimization
    parser.add_argument('--max_epoch', default=1, help='max epochs')
    parser.add_argument('--learning_rate', default=0.003, type=float, help='learning rate')
    parser.add_argument('--gama', default=0.5, type=float, help="margin parameter")
    parser.add_argument('--lam', default=0.1, type=float, help="trade-off parameter")
    parser.add_argument('--anomaly_ratio', default=0.05, type=float, help="anomaly ratio")
    parser.add_argument('--num_anomaly_num', default=300, type=int, help="number of anomalies")
    args = parser.parse_args()

    # data_name = args.dataset
    # model_name = args.model
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    dataset = Reader(args, args.data_path)
    if args.mode == 'train':
        train(args, dataset, device)
    elif args.mode == 'test':
        # raise NotImplementedError
        test(args, dataset, device)
    else:
        raise ValueError('Invalid mode')



def train(args, dataset, device):
    # Dataset parameters
    # data_name = args.dataset
    data_path = args.data_path
    model_name = args.model
    all_triples = dataset.train_data
    # labels = dataset.labels
    train_idx = list(range(len(all_triples) // 2))
    num_iterations = math.ceil(dataset.num_triples_with_anomalies / args.batch_size)
    total_num_anomalies = dataset.num_anomalies
    logging.basicConfig(level=logging.INFO)
    file_handler = logging.FileHandler(os.path.join(args.log_folder, model_name + "_" + args.dataset + "_" + str(args.anomaly_ratio)  + "_Neighbors" + str(args.num_neighbor) + "_" + "_log.txt"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logging.info('There are %d Triples with %d anomalies in the graph.' % (len(dataset.labels), total_num_anomalies))

    args.total_ent = dataset.num_entity
    args.total_rel = dataset.num_relation

    model_saved_path = model_name + "_" + args.dataset + "_" + str(args.anomaly_ratio) + ".ckpt"
    model_saved_path = os.path.join(args.save_dir, model_saved_path)
    # model.load_state_dict(torch.load(model_saved_path))
    # Model BiLSTM_Attention
    print("AAAAAAAAAA")
    model = BiLSTM_Attention(args, args.BiLSTM_input_size, args.BiLSTM_hidden_size, args.BiLSTM_num_layers, args.dropout,
                             args.alpha).to(device)
    print("BBBBBBBB")
    criterion = nn.MarginRankingLoss(args.gama)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    #
    for k in range(args.max_epoch):
        for it in range(num_iterations):
            # start_read_time = time.time()
            batch_h, batch_r, batch_t, batch_size = get_pair_batch_train_common(args, dataset, it, train_idx,
                                                                                args.batch_size,
                                                                                args.num_neighbor)
            # end_read_time = time.time()
            print("Time used in loading data", it)

            batch_h = torch.LongTensor(batch_h).to(device)
            batch_t = torch.LongTensor(batch_t).to(device)
            batch_r = torch.LongTensor(batch_r).to(device)

            out, out_att = model(batch_h, batch_r, batch_t)

            # running_time = time.time()
            # print("Time used in running model", math.fabs(end_read_time - running_time))

            out = out.reshape(batch_size, -1, 2 * 3 * args.BiLSTM_hidden_size)
            out_att = out_att.reshape(batch_size, -1, 2 * 3 * args.BiLSTM_hidden_size)

            pos_h = out[:, 0, :]
            pos_z0 = out_att[:, 0, :]
            pos_z1 = out_att[:, 1, :]
            neg_h = out[:, 1, :]
            neg_z0 = out_att[:, 2, :]
            neg_z1 = out_att[:, 3, :]

            # loss function
            # positive
            pos_loss = args.lam * torch.norm(pos_z0 - pos_z1, p=2, dim=1) + \
                       torch.norm(pos_h[:, 0:2 * args.BiLSTM_hidden_size] +
                                  pos_h[:, 2 * args.BiLSTM_hidden_size:2 * 2 * args.BiLSTM_hidden_size] -
                                  pos_h[:, 2 * 2 * args.BiLSTM_hidden_size:2 * 3 * args.BiLSTM_hidden_size], p=2,
                                  dim=1)
            # negative
            neg_loss = args.lam * torch.norm(neg_z0 - neg_z1, p=2, dim=1) + \
                       torch.norm(neg_h[:, 0:2 * args.BiLSTM_hidden_size] +
                                  neg_h[:, 2 * args.BiLSTM_hidden_size:2 * 2 * args.BiLSTM_hidden_size] -
                                  neg_h[:, 2 * 2 * args.BiLSTM_hidden_size:2 * 3 * args.BiLSTM_hidden_size], p=2,
                                  dim=1)

            y = -torch.ones(batch_size).to(device)
            loss = criterion(pos_loss, neg_loss, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pos_loss_value = torch.sum(pos_loss) / (batch_size * 2.0)
            neg_loss_value = torch.sum(neg_loss) / (batch_size * 2.0)
            logging.info('There are %d Triples in this batch.' % batch_size)
            logging.info('Epoch: %d-%d, pos_loss: %f, neg_loss: %f, Loss: %f' % (
                k, it + 1, pos_loss_value.item(), neg_loss_value.item(), loss.item()))

            # final_time = time.time()
            # print("BP time:", math.fabs(final_time - running_time))

            torch.save(model.state_dict(), model_saved_path)
    # # #
    # dataset = Reader(data_path, "test")



def test(args, dataset, device):
    # Dataset parameters
    # data_name = args.dataset
    data_path = args.data_path
    model_name = args.model
    all_triples = dataset.train_data
    # labels = dataset.labels
    train_idx = list(range(len(all_triples) // 2))
    num_iterations = math.ceil(dataset.num_triples_with_anomalies / args.batch_size)
    total_num_anomalies = dataset.num_anomalies
    logging.basicConfig(level=logging.INFO)
    file_handler = logging.FileHandler(os.path.join(args.log_folder, model_name + "_" + args.dataset + "_" + str(args.anomaly_ratio) + "_Neighbors" + str(args.num_neighbor) + "_" + "_log.txt"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logging.info('There are %d Triples with %d anomalies in the graph.' % (len(dataset.labels), total_num_anomalies))

    args.total_ent = dataset.num_entity
    args.total_rel = dataset.num_relation

    model_saved_path = model_name + "_" + args.dataset + "_" + str(args.anomaly_ratio) + ".ckpt"
    model_saved_path = os.path.join(args.save_dir, model_saved_path)

    model1 = BiLSTM_Attention(args, args.BiLSTM_input_size, args.BiLSTM_hidden_size, args.BiLSTM_num_layers, args.dropout,
                              args.alpha).to(device)
    model1.load_state_dict(torch.load(model_saved_path))
    model1.eval()
    with torch.no_grad():
        all_loss = []
        all_label = []
        all_pred = []
        start_id = 0
        # epochs = int(len(dataset.bp_triples_label) / 100)
        # 2720
        for i in range(num_iterations):
            # start_read_time = time.time()
            batch_h, batch_r, batch_t, labels, start_id, batch_size = get_pair_batch_test(dataset, args.batch_size,
                                                                                          args.num_neighbor, start_id)
            # labels = labels.unsqueeze(1)
            # batch_size = input_triples.size(0)


            batch_h = torch.LongTensor(batch_h).to(device)
            batch_t = torch.LongTensor(batch_t).to(device)
            batch_r = torch.LongTensor(batch_r).to(device)
            labels = labels.to(device)
            out, out_att = model1(batch_h, batch_r, batch_t)
            out_att = out_att.reshape(batch_size, 2, 2 * 3 * args.BiLSTM_hidden_size)
            out_att_view0 = out_att[:, 0, :]
            out_att_view1 = out_att[:, 1, :]
            # [B, 600] [B, 600]

            loss = args.lam * torch.norm(out_att_view0 - out_att_view1, p=2, dim=1) + \
                   torch.norm(out[:, 0:2 * args.BiLSTM_hidden_size] +
                              out[:, 2 * args.BiLSTM_hidden_size:2 * 2 * args.BiLSTM_hidden_size] -
                              out[:, 2 * 2 * args.BiLSTM_hidden_size:2 * 3 * args.BiLSTM_hidden_size], p=2, dim=1)

            all_loss += loss
            all_label += labels

            # print('{}th test data'.format(i))
            logging.info('[Train] Evaluation on %d batch of Original graph' % i)
            # sum = labels.sum()
            # if sum < labels.size(0):
            #     # loss = -1 * loss
            #     AUC = roc_auc_score(labels.cpu(), loss.cpu())
            #     print('AUC on the {}th test images: {} %'.format(i, np.around(AUC)))

        total_num = len(all_label)



        max_top_k = total_num_anomalies * 2
        min_top_k = total_num_anomalies // 10
        # all_loss = torch.from_numpy(np.array(list(all_loss.to(torch.device("cpu"))).astype(np.float))

        all_loss = np.array(all_loss)
        all_loss = torch.from_numpy(all_loss)

        top_loss, top_indices = torch.topk(all_loss, max_top_k, largest=True, sorted=True)
        top_labels = toarray([all_label[top_indices[iii]] for iii in range(len(top_indices))])

        anomaly_discovered = []
        for i in range(max_top_k):
            if i == 0:
                anomaly_discovered.append(top_labels[i])
            else:
                anomaly_discovered.append(anomaly_discovered[i-1] + top_labels[i])

        results_interval_10 = np.array([anomaly_discovered[i * 10] for i in range(max_top_k // 10)])
        # print(results_interval_10)
        logging.info('[Train] final results: %s' % str(results_interval_10))

        top_k = np.arange(1, max_top_k + 1)

        assert len(top_k) == len(anomaly_discovered), 'The size of result list is wrong'

        precision_k = np.array(anomaly_discovered) / top_k
        recall_k = np.array(anomaly_discovered) * 1.0 / total_num_anomalies

        precision_interval_10 = [precision_k[i * 10] for i in range(max_top_k // 10)]
        # print(precision_interval_10)
        logging.info('[Train] final Precision: %s' % str(precision_interval_10))
        recall_interval_10 = [recall_k[i * 10] for i in range(max_top_k // 10)]
        # print(recall_interval_10)
        logging.info('[Train] final Recall: %s' % str(recall_interval_10))

        logging.info('K = %d' % args.max_epoch)
        ratios = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.20, 0.30, 0.45]
        for i in range(len(ratios)):
            num_k = int(ratios[i] * dataset.num_original_triples)

            if num_k > len(anomaly_discovered):
                break

            recall = anomaly_discovered[num_k - 1] * 1.0 / total_num_anomalies
            precision = anomaly_discovered[num_k - 1] * 1.0 / num_k

            logging.info(
                '[Train][%s][%s] Precision %f -- %f : %f' % (args.dataset, model_name, args.anomaly_ratio, ratios[i], precision))
            logging.info('[Train][%s][%s] Recall  %f-- %f : %f' % (args.dataset, model_name, args.anomaly_ratio, ratios[i], recall))
            logging.info('[Train][%s][%s] anomalies in total: %d -- discovered:%d -- K : %d' % (
                args.dataset, model_name, total_num_anomalies, anomaly_discovered[num_k - 1], num_k))

if __name__ == '__main__':
    main()
