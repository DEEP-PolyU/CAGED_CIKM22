import numpy as np
import params
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
import time


def OurModel(ratio, kkkk, neighbors):
    # Dataset parameters
    data_path = params.data_dir_NELL_Triple
    params.num_neighbor = 9
    params.batch_size = 256
    # params.BiLSTM_input_size = 50
    # params.BiLSTM_hidden_size = 50
    # params.input_size_lstm = 50
    # params.hidden_size_lstm = 50
    # params.embedding_dim = 50
    # params.anomaly_ratio = ratio
    params.anomaly_ratio = ratio
    model_name = "Efficiency"
    data_name = "NELL_FULL_"
    dataset = Reader(data_path, isInjectTopK=False)
    all_triples = dataset.train_data
    labels = dataset.labels
    train_idx = list(range(len(all_triples) // 2))
    num_iterations = math.ceil(dataset.num_triples_with_anomalies / params.batch_size)
    total_num_anomalies = dataset.num_anomalies
    logging.basicConfig(level=logging.INFO)
    file_handler = logging.FileHandler(os.path.join(params.log_folder, model_name + "_" + data_name + "_" + str(ratio) + "_" + str(params.kkkkk) + "_Neighbors" + str(neighbors) + "_" + "_log.txt"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logging.info('There are %d Triples with %d anomalies in the graph.' % (len(dataset.labels), total_num_anomalies))

    params.total_ent = dataset.num_entity
    params.total_rel = dataset.num_relation

    model_saved_path = model_name + "_" + data_name + "_" + str(ratio) + ".ckpt"
    model_saved_path = os.path.join(params.out_folder, model_saved_path)
    # model.load_state_dict(torch.load(model_saved_path))
    # Model BiLSTM_Attention
    model = BiLSTM_Attention(params.input_size_lstm, params.hidden_size_lstm, params.num_layers_lstm, params.dropout,
                             params.alpha).to(params.device)
    criterion = nn.MarginRankingLoss(params.gama)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    #
    start_time = time.time()
    for k in range(1):
        for it in range(10):
            # start_read_time = time.time()
            batch_h, batch_r, batch_t, batch_size = get_pair_batch_train_common(dataset, it, train_idx,
                                                                                params.batch_size,
                                                                                params.num_neighbor)
            # end_read_time = time.time()
            # print("Time used in loading data", math.fabs(start_read_time - end_read_time))

            batch_h = torch.LongTensor(batch_h).to(params.device)
            batch_t = torch.LongTensor(batch_t).to(params.device)
            batch_r = torch.LongTensor(batch_r).to(params.device)
            # input_triple, batch_size = get_pair_batch_train_common(dataset, it, train_idx,
            #                                                                     params.batch_size,
            #                                                                     params.num_neighbor)
            # input_triple = Variable(torch.LongTensor(input_triple).cuda())
            # batch_size = input_triples.size(0)
            out, out_att = model(batch_h, batch_r, batch_t)

            # running_time = time.time()
            # print("Time used in running model", math.fabs(end_read_time - running_time))

            out = out.reshape(batch_size, -1, 2 * 3 * params.BiLSTM_hidden_size)
            out_att = out_att.reshape(batch_size, -1, 2 * 3 * params.BiLSTM_hidden_size)

            pos_h = out[:, 0, :]
            pos_z0 = out_att[:, 0, :]
            pos_z1 = out_att[:, 1, :]
            neg_h = out[:, 1, :]
            neg_z0 = out_att[:, 2, :]
            neg_z1 = out_att[:, 3, :]

            # loss function
            # positive
            pos_loss = params.lam * torch.norm(pos_z0 - pos_z1, p=2, dim=1) + \
                       torch.norm(pos_h[:, 0:2 * params.BiLSTM_hidden_size] +
                                  pos_h[:, 2 * params.BiLSTM_hidden_size:2 * 2 * params.BiLSTM_hidden_size] -
                                  pos_h[:, 2 * 2 * params.BiLSTM_hidden_size:2 * 3 * params.BiLSTM_hidden_size], p=2,
                                  dim=1)
            # negative
            neg_loss = params.lam * torch.norm(neg_z0 - neg_z1, p=2, dim=1) + \
                       torch.norm(neg_h[:, 0:2 * params.BiLSTM_hidden_size] +
                                  neg_h[:, 2 * params.BiLSTM_hidden_size:2 * 2 * params.BiLSTM_hidden_size] -
                                  neg_h[:, 2 * 2 * params.BiLSTM_hidden_size:2 * 3 * params.BiLSTM_hidden_size], p=2,
                                  dim=1)

            y = -torch.ones(batch_size).to(params.device)
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
    end_time = time.time()
    efficiency = (end_time - start_time) / 10
    print('Time: ', efficiency)
    # dataset = Reader(data_path, "test")
    model1 = BiLSTM_Attention(params.input_size_lstm, params.hidden_size_lstm, params.num_layers_lstm, params.dropout,
                              params.alpha).to(params.device)
    model1.load_state_dict(torch.load(model_saved_path))
    model1.eval()
    with torch.no_grad():
        all_loss = []
        all_label = []
        all_pred = []
        start_id = 0
        # epochs = int(len(dataset.bp_triples_label) / 100)
        # 2720
        num_iterations = 20
        for i in range(num_iterations):
            # start_read_time = time.time()
            batch_h, batch_r, batch_t, labels, start_id, batch_size = get_pair_batch_test(dataset, params.batch_size,
                                                                                          params.num_neighbor, start_id)
            # labels = labels.unsqueeze(1)
            # batch_size = input_triples.size(0)


            batch_h = torch.LongTensor(batch_h).to(params.device)
            batch_t = torch.LongTensor(batch_t).to(params.device)
            batch_r = torch.LongTensor(batch_r).to(params.device)
            labels = labels.to(params.device)
            out, out_att = model1(batch_h, batch_r, batch_t)
            out_att = out_att.reshape(batch_size, 2, 2 * 3 * params.BiLSTM_hidden_size)
            out_att_view0 = out_att[:, 0, :]
            out_att_view1 = out_att[:, 1, :]
            # [B, 600] [B, 600]

            loss = params.lam * torch.norm(out_att_view0 - out_att_view1, p=2, dim=1) + \
                   torch.norm(out[:, 0:2 * params.BiLSTM_hidden_size] +
                              out[:, 2 * params.BiLSTM_hidden_size:2 * 2 * params.BiLSTM_hidden_size] -
                              out[:, 2 * 2 * params.BiLSTM_hidden_size:2 * 3 * params.BiLSTM_hidden_size], p=2, dim=1)
            # pos = torch.ones(params.batch_size).to(params.device)
            # neg = torch.zeros(params.batch_size).to(params.device)
            # pred = torch.where(loss < 7., pos, neg)
            #
            # total += labels.size(0)
            # correct = (pred == labels).sum().item()
            # print('Test Accuracy of the model on the 100 test images: {} %'.format(1.0 * correct / labels.size(0)))
            # print(loss.shape)
            # print(labels.shape)
            # all_pred += pred
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

        # print("Total number of test tirples: ", total_num)
        # AUC11 = roc_auc_score(toarray(all_label), toarray_float(all_loss))
        # logging.info('[Train] AUC of %d triples: %f' % (total_num, AUC11))
        # print('AUC of {} triples: {}'.format(num_epochs * 100, np.around(AUC11, 4)))
        # correct = (toarray(all_pred) == toarray(all_label)).sum().item()
        # print('Accuracy of {} triples:{} %'.format(epochs * 100, 1 - 1.0 * correct / len(all_label)))
        # _, top1000_indices = torch.topk(toarray(all_loss), 1000, largest=True, sorted=False)

        # num_k = int(ratios[-1] * dataset.num_original_triples)
        # top_loss, top_indices = torch.topk(toarray_float(all_loss), num_k, largest=True, sorted=True)
        # for i in range(len(ratios)):
        #     num_k = int(ratios[i] * dataset.num_original_triples)
        #
        #     top_loss, top_indices = torch.topk(toarray_float(all_loss), num_k, largest=True, sorted=True)
        #     top_labels = toarray([all_label[top_indices[iii]] for iii in range(len(top_indices))])
        #     top_sum = top_labels.sum()
        #     recall = top_sum * 1.0 / total_num_anomalies
        #     precision = top_sum * 1.0 / num_k
        #
        #     logging.info('[Train][%s][%s] Precision %f -- %f : %f' % (data_name, model_name, ratio, ratios[i], precision))
        #     logging.info('[Train][%s][%s] Recall  %f-- %f : %f' % (data_name, model_name, ratio, ratios[i], recall))
        #     logging.info('[Train][%s][%s] anomalies in total: %d -- discovered:%d -- K : %d' % (
        #     data_name, model_name, total_num_anomalies, top_sum, num_k))
        #
        #     if top_sum.item() < num_k and top_sum.item() != 0:
        #         # print(top_sum.item(), num_k)
        #         AUC_K = roc_auc_score(toarray(top_labels), toarray_float(top_loss))
        #         logging.info('[Train][%s][%s] xxxxxxxxxxxxxxxxxxxxxxx AUC %f -- %f : %f' % (
        #             data_name, model_name, ratio, ratios[i], AUC_K))

        max_top_k = total_num_anomalies * 2
        min_top_k = total_num_anomalies // 10

        top_loss, top_indices = torch.topk(toarray_float(all_loss), max_top_k, largest=True, sorted=True)
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

        logging.info('K = %d' % kkkk)
        ratios = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
        for i in range(len(ratios)):
            num_k = int(ratios[i] * dataset.num_original_triples)

            if num_k > len(anomaly_discovered):
                break

            recall = anomaly_discovered[num_k - 1] * 1.0 / total_num_anomalies
            precision = anomaly_discovered[num_k - 1] * 1.0 / num_k

            logging.info(
                '[Train][%s][%s] Precision %f -- %f : %f' % (data_name, model_name, ratio, ratios[i], precision))
            logging.info('[Train][%s][%s] Recall  %f-- %f : %f' % (data_name, model_name, ratio, ratios[i], recall))
            logging.info('[Train][%s][%s] anomalies in total: %d -- discovered:%d -- K : %d' % (
                data_name, model_name, total_num_anomalies, anomaly_discovered[num_k - 1], num_k))

            # if top_sum.item() < num_k and top_sum.item() != 0:
            #     # print(top_sum.item(), num_k)
            #     AUC_K = roc_auc_score(toarray(top_labels), toarray_float(top_loss))
            #     logging.info('[Train][%s][%s] xxxxxxxxxxxxxxxxxxxxxxx AUC %f -- %f : %f' % (
            #         data_name, model_name, ratio, ratios[i], AUC_K))
        # plt.subplot(2, 1, 1)
        # plt.plot(top_k, anomaly_discovered)
        # plt.title('Anomaly Discovery Curve')
        #
        # plt.subplot(2, 1, 2)
        # plt.plot(recall_k, precision_k)
        # plt.title('Recall-Precision')
        # plt.show()

    # dataset1 = Reader(data_path, "test", isInjectTopK=False)
    # num_iterations = math.ceil(dataset1.num_triples_with_anomalies / params.batch_size)
    # total_num_anomalies = dataset1.num_anomalies
    # model1 = BiLSTM_Attention(params.input_size_lstm, params.hidden_size_lstm, params.num_layers_lstm, params.dropout,
    #                           params.alpha).to(params.device)
    # model1.load_state_dict(torch.load(model_saved_path))
    # model1.eval()
    # with torch.no_grad():
    #     all_loss = []
    #     all_label = []
    #     all_pred = []
    #     start_id = 0
    #
    #     for i in range(num_iterations):
    #         batch_h, batch_r, batch_t, labels, start_id, batch_size = get_pair_batch_test(dataset1, params.batch_size,
    #                                                                                       params.num_neighbor, start_id)
    #         # labels = labels.unsqueeze(1)
    #         # batch_size = input_triples.size(0)
    #         batch_h = torch.LongTensor(batch_h).to(params.device)
    #         batch_t = torch.LongTensor(batch_t).to(params.device)
    #         batch_r = torch.LongTensor(batch_r).to(params.device)
    #         labels = labels.to(params.device)
    #         out, out_att = model1(batch_h, batch_r, batch_t)
    #         out_att = out_att.reshape(batch_size, 2, 2 * 3 * params.BiLSTM_hidden_size)
    #         out_att_view0 = out_att[:, 0, :]
    #         out_att_view1 = out_att[:, 1, :]
    #         # [B, 600] [B, 600]
    #
    #         loss = params.lam * torch.norm(out_att_view0 - out_att_view1, p=2, dim=1) + \
    #                torch.norm(out[:, 0:2 * params.BiLSTM_hidden_size] +
    #                           out[:, 2 * params.BiLSTM_hidden_size:2 * 2 * params.BiLSTM_hidden_size] -
    #                           out[:, 2 * 2 * params.BiLSTM_hidden_size:2 * 3 * params.BiLSTM_hidden_size], p=2, dim=1)
    #         # pos = torch.ones(params.batch_size).to(params.device)
    #         # neg = torch.zeros(params.batch_size).to(params.device)
    #         # pred = torch.where(loss < 7., pos, neg)
    #         #
    #         # total += labels.size(0)
    #         # correct = (pred == labels).sum().item()
    #         # print('Test Accuracy of the model on the 100 test images: {} %'.format(1.0 * correct / labels.size(0)))
    #         # print(loss.shape)
    #         # print(labels.shape)
    #         # all_pred += pred
    #         all_loss += loss
    #         all_label += labels
    #
    #         # print('{}th test data'.format(i))
    #         logging.info('[Test] Evaluation on %d batch of Original graph' % i)
    #         # sum = labels.sum()
    #         # if sum < labels.size(0):
    #         #     # loss = -1 * loss
    #         #     AUC = roc_auc_score(labels.cpu(), loss.cpu())
    #         #     print('AUC on the {}th test images: {} %'.format(i, np.around(AUC)))
    #
    #     total_num = len(all_label)
    #
    #     # print("Total number of test tirples: ", total_num)
    #     AUC11 = roc_auc_score(toarray(all_label), toarray_float(all_loss))
    #     logging.info('[Test] AUC of %d triples: %f' % (total_num, AUC11))
    #     # print('AUC of {} triples: {}'.format(num_epochs * 100, np.around(AUC11, 4)))
    #     # correct = (toarray(all_pred) == toarray(all_label)).sum().item()
    #     # print('Accuracy of {} triples:{} %'.format(epochs * 100, 1 - 1.0 * correct / len(all_label)))
    #     # _, top1000_indices = torch.topk(toarray(all_loss), 1000, largest=True, sorted=False)
    #
    #     ratios = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
    #     for i in range(len(ratios)):
    #         num_k = int(ratios[i] * dataset1.num_original_triples)
    #         top_loss, top_indices = torch.topk(toarray_float(all_loss), num_k, largest=True, sorted=True)
    #         top_labels = toarray([all_label[top_indices[iii]] for iii in range(len(top_indices))])
    #         top_sum = top_labels.sum()
    #         recall = top_sum * 1.0 / total_num_anomalies
    #         precision = top_sum * 1.0 / num_k
    #
    #         logging.info('[Test][%s][%s] Precision %f -- %f : %f' % (data_name, model_name, ratio, ratios[i], precision))
    #         logging.info('[Test][%s][%s] Recall  %f-- %f : %f' % (data_name, model_name, ratio, ratios[i], recall))
    #         logging.info('[Test][%s][%s] anomalies in total: %d -- discovered:%d -- K : %d' % (
    #             data_name, model_name, total_num_anomalies, top_sum, num_k))
    #
    #         if top_sum.item() < num_k and top_sum.item() != 0:
    #             # print(top_sum.item(), num_k)
    #             AUC_K = roc_auc_score(toarray(top_labels), toarray_float(top_loss))
    #             logging.info('[Test][%s][%s] xxxxxxxxxxxxxxxxxxxxxxx AUC %f -- %f : %f' % (
    #                 data_name, model_name, ratio, ratios[i], AUC_K))
    #
    #     max_top_k = total_num_anomalies * 2
    #     min_top_k = total_num_anomalies // 10
    #
    #     top_loss, top_indices = torch.topk(toarray_float(all_loss), max_top_k, largest=True, sorted=True)
    #     top_labels = toarray([all_label[top_indices[iii]] for iii in range(len(top_indices))])
    #
    #     anomaly_discovered = []
    #     for i in range(max_top_k):
    #         if i == 0:
    #             anomaly_discovered.append(top_labels[i])
    #         else:
    #             anomaly_discovered.append(anomaly_discovered[i - 1] + top_labels[i])
    #
    #     results_interval_10 = np.array([anomaly_discovered[i * 10] for i in range(max_top_k // 10)])
    #     # print(results_interval_10)
    #     logging.info('[Test] final results: %s' % str(results_interval_10))
    #
    #     top_k = np.arange(1, max_top_k + 1)
    #
    #     assert len(top_k) == len(anomaly_discovered), 'The size of result list is wrong'
    #
    #     precision_k = np.array(anomaly_discovered) / top_k
    #     recall_k = np.array(anomaly_discovered) * 1.0 / total_num_anomalies
    #
    #     precision_interval_10 = [precision_k[i * 10] for i in range(max_top_k // 10)]
    #     # print(precision_interval_10)
    #     logging.info('[Test] final Precision: %s' % str(precision_interval_10))
    #     recall_interval_10 = [recall_k[i * 10] for i in range(max_top_k // 10)]
    #     # print(recall_interval_10)
    #     logging.info('[Test] final Recall: %s' % str(recall_interval_10))


anomaly_injected_ratios = [0.01, 0.05, 0.10, 0.15]
num_epochs = [3]
# anomaly_injected_ratios = [0.01, 0.05, 0.10, 0.01075235, 0.021505, 0.032258]
# num_epochs = [1]
neighbors = [19]

for num in num_epochs:
    for ratio in anomaly_injected_ratios:
        for n in neighbors:
            OurModel(ratio, num, n)
