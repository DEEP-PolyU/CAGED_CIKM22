import torch
import numpy as np
import random
from random import shuffle

def toarray(x):
    return torch.from_numpy(np.array(list(x)).astype(np.int32))

def toarray_float(x):
    return torch.from_numpy(np.array(list(x)).astype(np.float))

def get_neighbor_id(ent, h2t, t2h, A):
    hrt = []
    hrt1 = []
    hrt2 = []
    if ent in h2t.keys():
        tails = list(h2t[ent])
        # print('tails', tails)
        # for i in range(len(tails)):
        #     hrt1.append((ent, A[ent][tails[i]], tails[i]))
        hrt1 = [(ent, A[(ent, i)], i) for i in tails]

        # print('hrt1', hrt1)

    if ent in t2h.keys():
        heads = list(t2h[ent])
        # for i in range(len(heads)):
        #     hrt2.append((heads[i], A[heads[i]][ent], ent))
        hrt2 = [(i, A[(i, ent)], ent) for i in heads]
        # print('hrt2', hrt2)

    hrt = hrt1 + hrt2

    return hrt


def get_triple_neighbor(h, r, t, dataset, num_neighbor):
    h_neighbor = 0
    h2t = dataset.h2t
    t2h = dataset.t2h
    A = dataset.A

    # print('h, r, t', h, r, t)
    head_neighbor = get_neighbor_id(h, h2t, t2h, A)
    tail_neighbor = get_neighbor_id(t, h2t, t2h, A)
    # hrt_neighbor = get_neighbor_id(h, h2t, t2h, A) + get_neighbor_id(t, h2t, t2h, A)
    # while (h, r, t) in hrt_neighbor:
    #     hrt_neighbor.remove((h, r, t))
    # if len(head_neighbor) == 0:
    #     temp = [(h, r, t)]
    #     hh_neighbors = random.choices(temp, k=num_neighbor)
    if len(head_neighbor) > num_neighbor:
        hh_neighbors = random.sample(head_neighbor, k=num_neighbor)
    elif len(head_neighbor) > 0:
        hh_neighbors = random.choices(head_neighbor, k=num_neighbor)
    else:
        temp = [(h, r, t)]
        hh_neighbors = random.choices(temp, k=num_neighbor)

    if len(tail_neighbor) > num_neighbor:
        tt_neighbors = random.sample(tail_neighbor, k=num_neighbor)
    elif num_neighbor > len(tail_neighbor):
        if len(tail_neighbor) > 0:
            tt_neighbors = random.choices(tail_neighbor, k=num_neighbor)
        else:
            # print('hrt=null', len(hrt_neighbor))
            temp = [(h, r, t)]
            tt_neighbors = random.choices(temp, k=num_neighbor)
    else:
        tt_neighbors = tail_neighbor

    hrt_neighbor = [(h, r, t)] + hh_neighbors + [(h, r, t)] + tt_neighbors
    # print('hrt_neighbor', len(hrt_neighbor))
    # print("hn, tn", (len(head_neighbor), len(tail_neighbor)))


    return hrt_neighbor


def get_triple_batch(args, dataset, batch_size, num_neighbor):
    h, r, t, all_triples, labels, A = dataset.next_batch()
    labels = labels.unsqueeze(1)
    all_triples_labels = torch.cat((all_triples, labels), dim=1)
    print('all_triples_labels.shape', all_triples_labels.shape)

    sample_triple_labels = random.sample(list(all_triples_labels), k=batch_size)
    batch_triples = []
    batch_labels = []
    for i in range(batch_size):
        hrt_neighbor = get_triple_neighbor(sample_triple_labels[i][0].item(), sample_triple_labels[i][1].item(), sample_triple_labels[i][2].item(), dataset, num_neighbor)
        batch_triples = batch_triples + hrt_neighbor
        batch_labels.append(sample_triple_labels[i][3])

    print('batch_triples', batch_triples)
    ent_vec, rel_vec = dataset.ent_vec, dataset.rel_vec

    head_embedding = np.array([ent_vec[batch_triples[i][0]] for i in range(len(batch_triples))])
    head_embedding = torch.from_numpy(head_embedding)

    relation_embedding = np.array([rel_vec[batch_triples[i][1]] for i in range(len(batch_triples))])
    relation_embedding = torch.from_numpy(relation_embedding)

    tail_embedding = np.array([ent_vec[batch_triples[i][2]] for i in range(len(batch_triples))])
    tail_embedding = torch.from_numpy(tail_embedding)

    batch_triples_emb = torch.cat((head_embedding, relation_embedding), dim=1)
    batch_triples_emb = torch.cat((batch_triples_emb, tail_embedding), dim=1)

    batch_triples_emb = batch_triples_emb.view(-1, 3, args.BiLSTM_input_size)
    print('input_batch.shape:', batch_triples_emb.shape)

    return batch_triples_emb, toarray(batch_triples), toarray(batch_labels)


def get_batch_only(dataset, batch_size, start_id):
    triples, labels = dataset.get_data_embed()

    if start_id + batch_size >= len(triples):
        batch_triples = triples[start_id:]
        batch_labels = labels[start_id:]
        start_id = 0
    else:
        batch_triples = triples[start_id: start_id + batch_size]
        batch_labels = labels[start_id: start_id + batch_size]
        start_id += batch_size

    return batch_triples, batch_labels, start_id


def get_triple_pair(dataset, batch_size, num_neighbor):
    h, r, t, all_triples, labels, A = dataset.get_data()
    labels = labels.unsqueeze(1)
    all_triples_labels = torch.cat((all_triples, labels), dim=1)
    # print('all_triples_labels.shape', all_triples_labels.shape)

    n = all_triples_labels.size(0) // 2
    idx = random.randint(0, n - 1)
    sample_triple_labels = [all_triples_labels[idx], all_triples_labels[idx + n]]
    batch_triples = []
    batch_labels = []

    for i in range(batch_size*2):
        hrt_neighbor = get_triple_neighbor(sample_triple_labels[i][0].item(), sample_triple_labels[i][1].item(), sample_triple_labels[i][2].item(), dataset, num_neighbor)
        batch_triples = batch_triples + hrt_neighbor
        batch_labels.append(sample_triple_labels[i][3])

    # print('batch_triples', batch_triples)
    ent_vec, rel_vec = dataset.ent_vec, dataset.rel_vec

    head_embedding = np.array([ent_vec[batch_triples[i][0]] for i in range(len(batch_triples))])
    head_embedding = torch.from_numpy(head_embedding)

    relation_embedding = np.array([rel_vec[batch_triples[i][1]] for i in range(len(batch_triples))])
    relation_embedding = torch.from_numpy(relation_embedding)

    tail_embedding = np.array([ent_vec[batch_triples[i][2]] for i in range(len(batch_triples))])
    tail_embedding = torch.from_numpy(tail_embedding)

    batch_triples_emb = torch.cat((head_embedding, relation_embedding), dim=1)
    batch_triples_emb = torch.cat((batch_triples_emb, tail_embedding), dim=1)

    batch_triples_emb = batch_triples_emb.view(-1, 3, 100)
    # print('input_batch.shape:', batch_triples_emb.shape)

    return batch_triples_emb, toarray(batch_triples), toarray(batch_labels)


def get_pair_batch_train(args, dataset, batch_size, num_neighbor):
    all_triples = dataset.train_data
    # all_triples_labels = torch.cat((all_triples, labels), dim=1)
    # print('all_triples_labels.shape', all_triples_labels.shape)

    n = all_triples.size(0) // 2
    sample_triple = []
    for i in range(batch_size):
        idx = random.randint(0, n-1)
        temp = [all_triples[idx], all_triples[idx + n]]
        sample_triple += temp
    # idx = random.randint(0, n - 1)
    # sample_triple_labels = [all_triples_labels[idx], all_triples_labels[idx + n]]
    batch_triples = []
    for i in range(batch_size*2):
        hrt_neighbor = get_triple_neighbor(sample_triple[i][0].item(), sample_triple[i][1].item(), sample_triple[i][2].item(), dataset, num_neighbor)
        batch_triples = batch_triples + hrt_neighbor

    # print('batch_triples', batch_triples)
    ent_vec, rel_vec = dataset.ent_vec, dataset.rel_vec

    head_embedding = np.array([ent_vec[batch_triples[i][0]] for i in range(len(batch_triples))])
    head_embedding = torch.from_numpy(head_embedding)

    relation_embedding = np.array([rel_vec[batch_triples[i][1]] for i in range(len(batch_triples))])
    relation_embedding = torch.from_numpy(relation_embedding)

    tail_embedding = np.array([ent_vec[batch_triples[i][2]] for i in range(len(batch_triples))])
    tail_embedding = torch.from_numpy(tail_embedding)

    batch_triples_emb = torch.cat((head_embedding, relation_embedding), dim=1)
    batch_triples_emb = torch.cat((batch_triples_emb, tail_embedding), dim=1)

    batch_triples_emb = batch_triples_emb.view(-1, 3, args.BiLSTM_input_size)
    # print('input_batch.shape:', batch_triples_emb.shape)

    return batch_triples_emb, toarray(batch_triples)


# def get_pair_batch_test(dataset, batch_size, num_neighbor, start_id):
#     # all_triples, labels, A = dataset.get_data_test()
#     all_triples = dataset.triples_with_anomalies
#     labels = dataset.triples_with_anomalies_labels
#     # all_triples_labels = torch.cat((all_triples, labels), dim=1)
#     print('all_triples.shape', all_triples.shape)
#
#     labels = labels.unsqueeze(1)
#
#     length = batch_size
#     if start_id + batch_size > len(all_triples):
#         sample_triple = all_triples[start_id:]
#         batch_labels = labels[start_id:]
#         length = len(sample_triple)
#         start_id = 0
#
#     else:
#         sample_triple = all_triples[start_id: start_id + batch_size]
#         batch_labels = labels[start_id: start_id + batch_size]
#         start_id += batch_size
#
#     batch_triples = []
#     # print('sample_triple.shape', sample_triple.shape)
#     for i in range(len(sample_triple)):
#         hrt_neighbor = get_triple_neighbor(sample_triple[i][0].item(), sample_triple[i][1].item(), sample_triple[i][2].item(), dataset, num_neighbor)
#         batch_triples = batch_triples + hrt_neighbor
#
#     # print('batch_triples.shape', toarray(batch_triples).shape)
#     ent_vec, rel_vec = dataset.ent_vec, dataset.rel_vec
#     # print("ent_vec.size", toarray(ent_vec).shape)
#     # print("rel_vec.shape", toarray(rel_vec).shape)
#     # assert ent_vec.size()[1] == args.BiLSTM_input_size, 'dimension of the input is not correct!'
#     head_embedding = np.array([ent_vec[batch_triples[i][0]] for i in range(len(batch_triples))])
#     head_embedding = torch.from_numpy(head_embedding)
#
#     relation_embedding = np.array([rel_vec[batch_triples[i][1]] for i in range(len(batch_triples))])
#     relation_embedding = torch.from_numpy(relation_embedding)
#
#     tail_embedding = np.array([ent_vec[batch_triples[i][2]] for i in range(len(batch_triples))])
#     tail_embedding = torch.from_numpy(tail_embedding)
#
#     batch_triples_emb = torch.cat((head_embedding, relation_embedding), dim=1)
#     batch_triples_emb = torch.cat((batch_triples_emb, tail_embedding), dim=1)
#
#     batch_triples_emb = batch_triples_emb.view(-1, 3, args.BiLSTM_input_size)
#     # print('input_batch.shape:', batch_triples_emb.shape)
#
#     return batch_triples_emb, toarray(batch_triples), toarray(batch_labels), start_id, length


def get_batch_baseline(args, all_triples, train_idx, n_batch):

    if n_batch == 0:
        np.random.shuffle(train_idx)
    if (n_batch + 1) * args.batch_size > len(train_idx):
        ids = train_idx[n_batch * args.batch_size:]
        # length = len(train_idx) - n_batch * args.batch_size
    else:
        ids = train_idx[n_batch * args.batch_size: (n_batch + 1) * args.batch_size]

    # train_pos = all_triples[0:len(all_triples) // 2, :]
    # train_neg = all_triples[len(all_triples) // 2:, :]

    batch_h = np.append(np.array([all_triples[i][0] for i in ids]),
                        np.array([all_triples[i + len(all_triples) // 2][0] for i in ids]))
    batch_r = np.append(np.array([all_triples[i][1] for i in ids]),
                        np.array([all_triples[i + len(all_triples) // 2][1] for i in ids]))
    batch_t = np.append(np.array([all_triples[i][2] for i in ids]),
                        np.array([all_triples[i + len(all_triples) // 2][2] for i in ids]))

    batch_y = np.concatenate((np.ones(args.batch_size), -1 * np.ones(args.batch_size)))

    return batch_h, batch_t, batch_r, batch_y
# return toarray(batch_h), toarray(batch_t), toarray(batch_r), toarray(batch_y)


def get_batch_baseline_test(args, all_triples, labels, train_idx, n_batch):
    if n_batch == 0:
        np.random.shuffle(train_idx)
    if (n_batch + 1) * args.batch_size > len(train_idx):
        ids = train_idx[n_batch * args.batch_size:]
        # length = len(train_idx) - n_batch * args.batch_size
    else:
        ids = train_idx[n_batch * args.batch_size: (n_batch + 1) * args.batch_size]
    # ids = train_idx[n_batch * args.batch_size: (n_batch + 1) * args.batch_size]

    batch_h = np.append(np.array([all_triples[i][0] for i in ids]),
                        np.array([all_triples[i + len(all_triples) // 2][0] for i in ids]))
    batch_r = np.append(np.array([all_triples[i][1] for i in ids]),
                        np.array([all_triples[i + len(all_triples) // 2][1] for i in ids]))
    batch_t = np.append(np.array([all_triples[i][2] for i in ids]),
                        np.array([all_triples[i + len(all_triples) // 2][2] for i in ids]))

    batch_y = np.concatenate((np.ones(args.batch_size), -1 * np.ones(args.batch_size)))
    label = np.array([labels[i] for i in ids])

    return batch_h, batch_t, batch_r, batch_y, label


def get_pair_batch_train_common(args, dataset, n_batch, train_idx, batch_size, num_neighbor):
    all_triples = dataset.train_data
    # all_triples_labels = torch.cat((all_triples, labels), dim=1)
    # print('all_triples_labels.shape', all_triples_labels.shape)
    if n_batch == 0:
        np.random.shuffle(train_idx)
    length = batch_size
    if (n_batch + 1) * batch_size > len(train_idx):
        ids = train_idx[n_batch * batch_size:]
        length = len(train_idx) - n_batch * batch_size
    else:
        ids = train_idx[n_batch * batch_size: (n_batch + 1) * batch_size]

    assert length == len(ids), "ERROR: batch_size != length."

    n = all_triples.size(0) // 2
    sample_triple = []
    for i in range(len(ids)):
        idx = ids[i]
        temp = [all_triples[idx], all_triples[idx + n]]
        sample_triple += temp
    # idx = random.randint(0, n - 1)
    # sample_triple_labels = [all_triples_labels[idx], all_triples_labels[idx + n]]
    batch_triples = []
    for i in range(len(sample_triple)):
        hrt_neighbor = get_triple_neighbor(sample_triple[i][0].item(), sample_triple[i][1].item(), sample_triple[i][2].item(), dataset, num_neighbor)
        batch_triples = batch_triples + hrt_neighbor

    batch_h = np.array([batch_triples[i][0] for i in range(len(batch_triples))])
    batch_r = np.array([batch_triples[i][1] for i in range(len(batch_triples))])
    batch_t = np.array([batch_triples[i][2] for i in range(len(batch_triples))])

    return batch_h, batch_r, batch_t, length



def get_pair_batch_test(dataset, batch_size, num_neighbor, start_id):
    # all_triples, labels, A = dataset.get_data_test()
    all_triples = dataset.triples_with_anomalies
    labels = dataset.triples_with_anomalies_labels
    # all_triples_labels = torch.cat((all_triples, labels), dim=1)
    # print('all_triples_labels.shape', all_triples_labels.shape)
    labels = labels.unsqueeze(1)

    length = batch_size
    if start_id + batch_size > len(all_triples):
        sample_triple = all_triples[start_id:]
        batch_labels = labels[start_id:]
        length = len(sample_triple)
        start_id = 0

    else:
        sample_triple = all_triples[start_id: start_id + batch_size]
        batch_labels = labels[start_id: start_id + batch_size]
        start_id += batch_size

    batch_triples = []
    for i in range(len(sample_triple)):
        hrt_neighbor = get_triple_neighbor(sample_triple[i][0].item(), sample_triple[i][1].item(), sample_triple[i][2].item(), dataset, num_neighbor)
        batch_triples = batch_triples + hrt_neighbor

    batch_h = np.array([batch_triples[i][0] for i in range(len(batch_triples))])
    batch_r = np.array([batch_triples[i][1] for i in range(len(batch_triples))])
    batch_t = np.array([batch_triples[i][2] for i in range(len(batch_triples))])

    return batch_h, batch_r, batch_t, toarray(batch_labels), start_id, length
    # print('batch_triples', batch_triples)
    # ent_vec, rel_vec = dataset.ent_vec, dataset.rel_vec
    #
    # head_embedding = np.array([ent_vec[batch_triples[i][0]] for i in range(len(batch_triples))])
    # head_embedding = torch.from_numpy(head_embedding)
    #
    # relation_embedding = np.array([rel_vec[batch_triples[i][1]] for i in range(len(batch_triples))])
    # relation_embedding = torch.from_numpy(relation_embedding)
    #
    # tail_embedding = np.array([ent_vec[batch_triples[i][2]] for i in range(len(batch_triples))])
    # tail_embedding = torch.from_numpy(tail_embedding)
    #
    # batch_triples_emb = torch.cat((head_embedding, relation_embedding), dim=1)
    # batch_triples_emb = torch.cat((batch_triples_emb, tail_embedding), dim=1)
    #
    # batch_triples_emb = batch_triples_emb.view(-1, 3, args.BiLSTM_input_size)
    # # print('input_batch.shape:', batch_triples_emb.shape)
    #
    # return batch_triples_emb, toarray(batch_triples), toarray(batch_labels), start_id, length
# def get_pair_batch_train_common(dataset, n_batch, train_idx, batch_size, num_neighbor):
#     all_triples = dataset.train_data
#     # all_triples_labels = torch.cat((all_triples, labels), dim=1)
#     # print('all_triples_labels.shape', all_triples_labels.shape)
#     if n_batch == 0:
#         np.random.shuffle(train_idx)
#     length = batch_size
#     if (n_batch + 1) * batch_size > len(train_idx):
#         ids = train_idx[n_batch * batch_size:]
#         length = len(train_idx) - n_batch * batch_size
#     else:
#         ids = train_idx[n_batch * batch_size: (n_batch + 1) * batch_size]
#
#     assert length == len(ids), "ERROR: batch_size != length."
#
#     n = all_triples.size(0) // 2
#     sample_triple = []
#     for i in range(len(ids)):
#         idx = ids[i]
#         temp = [all_triples[idx], all_triples[idx + n]]
#         sample_triple += temp
#     # idx = random.randint(0, n - 1)
#     # sample_triple_labels = [all_triples_labels[idx], all_triples_labels[idx + n]]
#     batch_triples = []
#     for i in range(len(sample_triple)):
#         hrt_neighbor = get_triple_neighbor(sample_triple[i][0].item(), sample_triple[i][1].item(), sample_triple[i][2].item(), dataset, num_neighbor)
#         batch_triples = batch_triples + hrt_neighbor
#     #     print("hrt_neighbor111:", len(hrt_neighbor))
#     # print("num of batch_triples", len(batch_triples))
#     # print("shape of batchtriples", batch_triples.shape())
#
#     # print('batch_triples', batch_triples)
#     ent_vec, rel_vec = dataset.ent_vec, dataset.rel_vec
#
#     head_embedding = np.array([ent_vec[batch_triples[i][0]] for i in range(len(batch_triples))])
#     head_embedding = torch.from_numpy(head_embedding)
#
#     relation_embedding = np.array([rel_vec[batch_triples[i][1]] for i in range(len(batch_triples))])
#     relation_embedding = torch.from_numpy(relation_embedding)
#
#     tail_embedding = np.array([ent_vec[batch_triples[i][2]] for i in range(len(batch_triples))])
#     tail_embedding = torch.from_numpy(tail_embedding)
#
#     batch_triples_emb = torch.cat((head_embedding, relation_embedding), dim=1)
#     batch_triples_emb = torch.cat((batch_triples_emb, tail_embedding), dim=1)
#
#     batch_triples_emb = batch_triples_emb.view(-1, 3, args.BiLSTM_input_size)
#     # print('input_batch.shape:', batch_triples_emb.shape)
#
#     return batch_triples_emb, toarray(batch_triples), length

