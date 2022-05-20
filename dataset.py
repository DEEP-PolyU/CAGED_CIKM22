import numpy as np
import random
import torch
import math
import params
from random import shuffle


class Reader:
    def __init__(self, path, isInjectTopK):
        self.isInjectTopK = isInjectTopK

        self.ent2id = dict()
        self.rel2id = dict()
        self.id2ent = dict()
        self.id2rel = dict()
        self.h2t = {}
        self.t2h = {}

        self.num_anomalies = 0
        self.triples = []
        self.start_batch = 0
        self.path = path

        self.A = {}
        # self.ent_vec, self.rel_vec = self.init_embeddings(path + '/' + params.dir_emb_ent,
        #                                                   self.path + '/' + params.dir_emb_rel)
        # self.num_entity = len(self.ent_vec)
        # self.num_relation = len(self.rel_vec)
        # self.A = np.ones((len(self.ent_vec), len(self.ent_vec)))
        # self.A = -1 * self.A
        # self.A = self.A.astype(np.int32)
        # print('A.size', self.A.shape, self.A[100][100])
        if self.path == params.data_dir_YAGO or self.path == params.data_dir_NELL or self.path == params.data_dir_DBPEDIA:
            self.read_triples_yago3()
        else:
            self.read_triples()
        self.triple_ori_set = set(self.triples)
        self.num_original_triples = len(self.triples)

        self.num_entity = self.num_ent()
        self.num_relation = self.num_rel()
        print('entity&relation: ', self.num_entity, self.num_relation)

        self.bp_triples_label = self.inject_anomaly()

        self.num_triples_with_anomalies = len(self.bp_triples_label)
        self.train_data, self.labels = self.get_data()
        self.triples_with_anomalies, self.triples_with_anomalies_labels = self.get_data_test()

    # def train_triples(self):
    #     return self.triples["train"]
    #
    # def valid_triples(self):
    #     return self.triples["valid"]
    #
    # def test_triples(self):
    #     return self.triples["test"]

    # def all_triples(self):
    #     return self.triples["train"] + self.triples["valid"] + self.triples["test"]

    def num_ent(self):
        return len(self.ent2id)

    def num_rel(self):
        return len(self.rel2id)

    def get_add_ent_id(self, ent):
        if ent in self.ent2id:
            ent_id = self.ent2id[ent]
        else:
            ent_id = len(self.ent2id)
            self.ent2id[ent] = ent_id
            self.id2ent[ent_id] = ent

        return ent_id

    def get_add_rel_id(self, rel):
        if rel in self.rel2id:
            rel_id = self.rel2id[rel]
        else:
            rel_id = len(self.rel2id)
            self.rel2id[rel] = rel_id
            self.id2rel[rel_id] = rel
        return rel_id

    def init_embeddings(self, entity_file, relation_file):
        entity_emb, relation_emb = [], []

        with open(entity_file) as f:
            for line in f:
                entity_emb.append([float(val) for val in line.strip().split()])

        with open(relation_file) as f:
            for line in f:
                relation_emb.append([float(val) for val in line.strip().split()])

        return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)

    def read_triples(self):
        print('Read begin!')
        for file in ["train", "valid", "test"]:
            with open(self.path + '/' + file + ".txt", "r") as f:
                for line in f.readlines():
                    try:
                        head, rel, tail = line.strip().split("\t")
                    except:
                        print(line)
                    head_id = self.get_add_ent_id(head)
                    rel_id = self.get_add_rel_id(rel)
                    tail_id = self.get_add_ent_id(tail)

                    self.triples.append((head_id, rel_id, tail_id))

                    self.A[(head_id, tail_id)] = rel_id
                    # self.A[head_id][tail_id] = rel_id

                    # generate h2t
                    if not head_id in self.h2t.keys():
                        self.h2t[head_id] = set()
                    temp = self.h2t[head_id]
                    temp.add(tail_id)
                    self.h2t[head_id] = temp

                    # generate t2h
                    if not tail_id in self.t2h.keys():
                        self.t2h[tail_id] = set()
                    temp = self.t2h[tail_id]
                    temp.add(head_id)
                    self.t2h[tail_id] = temp

        print("Read end!")
        return self.triples

    def read_triples_yago3(self):
        print('Read begin!')
        for file in ["train", "valid", "test"]:
            with open(self.path + '/' + file + ".txt", "r", encoding="utf-8") as f:
                train = f.readlines()
                # train_ = set({})
                for i in range(len(train)):
                    x = train[i].split()
                    x_ = tuple(x)
                    head, rel, tail = x_[0], x_[1], x_[2]

                    head_id = self.get_add_ent_id(head)
                    rel_id = self.get_add_rel_id(rel)
                    tail_id = self.get_add_ent_id(tail)

                    self.triples.append((head_id, rel_id, tail_id))

                    self.A[(head_id, tail_id)] = rel_id
                    # self.A[head_id][tail_id] = rel_id

                    # generate h2t
                    if not head_id in self.h2t.keys():
                        self.h2t[head_id] = set()
                    temp = self.h2t[head_id]
                    temp.add(tail_id)
                    self.h2t[head_id] = temp

                    # generate t2h
                    if not tail_id in self.t2h.keys():
                        self.t2h[tail_id] = set()
                    temp = self.t2h[tail_id]
                    temp.add(head_id)
                    self.t2h[tail_id] = temp

                del (train)

        print("Read end!")
        return self.triples

    def rand_ent_except(self, ent):
        rand_ent = random.randint(1, self.num_ent() - 1)
        while rand_ent == ent:
            rand_ent = random.randint(1, self.num_ent() - 1)
        return rand_ent

    def generate_neg_triples(self, pos_triples):
        neg_triples = []
        for head, rel, tail in pos_triples:
            head_or_tail = random.randint(0, 1)
            if head_or_tail == 0:
                new_head = self.rand_ent_except(head)
                neg_triples.append((new_head, rel, tail))
            else:
                new_tail = self.rand_ent_except(tail)
                neg_triples.append((head, rel, new_tail))
        return neg_triples

    def generate_anomalous_triples(self, pos_triples):
        neg_triples = []
        for head, rel, tail in pos_triples:
            head_or_tail = random.randint(0, 2)
            if head_or_tail == 0:
                new_head = random.randint(0, self.num_entity - 1)
                new_relation = rel
                new_tail = tail
                # neg_triples.append((new_head, rel, tail))
            elif head_or_tail == 1:
                new_head = head
                new_relation = random.randint(0, self.num_relation - 1)
                new_tail = tail
            else:
                # new_tail = self.rand_ent_except(tail)
                # neg_triples.append((head, rel, new_tail))
                new_head = head
                new_relation = rel
                new_tail = random.randint(0, self.num_entity - 1)
            anomaly = (new_head, new_relation, new_tail)
            while anomaly in self.triple_ori_set:
                if head_or_tail == 0:
                    new_head = random.randint(0, self.num_entity - 1)
                    new_relation = rel
                    new_tail = tail
                    # neg_triples.append((new_head, rel, tail))
                elif head_or_tail == 1:
                    new_head = head
                    new_relation = random.randint(0, self.num_relation - 1)
                    new_tail = tail
                else:
                    # new_tail = self.rand_ent_except(tail)
                    # neg_triples.append((head, rel, new_tail))
                    new_head = head
                    new_relation = rel
                    new_tail = random.randint(0, self.num_entity - 1)
                anomaly = (new_head, new_relation, new_tail)
            neg_triples.append(anomaly)
        return neg_triples

    def generate_anomalous_triples_2(self, num_anomaly):
        neg_triples = []
        for i in range(num_anomaly):
            new_head = random.randint(0, self.num_entity - 1)
            new_relation = random.randint(0, self.num_relation - 1)
            new_tail = random.randint(0, self.num_entity - 1)

            anomaly = (new_head, new_relation, new_tail)

            while anomaly in self.triple_ori_set:
                new_head = random.randint(0, self.num_entity - 1)
                new_relation = random.randint(0, self.num_relation - 1)
                new_tail = random.randint(0, self.num_entity - 1)
                anomaly = (new_head, new_relation, new_tail)

            neg_triples.append(anomaly)
        return neg_triples

    def shred_triples(self, triples):
        h_dix = [triples[i][0] for i in range(len(triples))]
        r_idx = [triples[i][1] for i in range(len(triples))]
        t_idx = [triples[i][2] for i in range(len(triples))]
        return h_dix, r_idx, t_idx

    def shred_triples_and_labels(self, triples_and_labels):
        heads = [triples_and_labels[i][0][0] for i in range(len(triples_and_labels))]
        rels = [triples_and_labels[i][0][1] for i in range(len(triples_and_labels))]
        tails = [triples_and_labels[i][0][2] for i in range(len(triples_and_labels))]
        labels = [triples_and_labels[i][1] for i in range(len(triples_and_labels))]
        return heads, rels, tails, labels

    def all_triplets(self):
        ph_all, pr_all, pt_all = self.shred_triples(self.triples)
        nh_all, nr_all, nt_all = self.shred_triples(self.generate_neg_triples(self.triples))
        return ph_all, pt_all, nh_all, nt_all, pr_all

    def get_data(self):
        # bp_triples_label = self.inject_anomaly()
        bp_triples_label = self.bp_triples_label
        labels = [bp_triples_label[i][1] for i in range(len(bp_triples_label))]
        bp_triples = [bp_triples_label[i][0] for i in range(len(bp_triples_label))]
        bn_triples = self.generate_anomalous_triples(bp_triples)
        all_triples = bp_triples + bn_triples

        return self.toarray(all_triples), self.toarray(labels)

    def get_data_test(self):
        bp_triples_label = self.bp_triples_label
        labels = [bp_triples_label[i][1] for i in range(len(bp_triples_label))]
        bp_triples = [bp_triples_label[i][0] for i in range(len(bp_triples_label))]

        return self.toarray(bp_triples), self.toarray(labels)

    def toarray(self, x):
        return torch.from_numpy(np.array(list(x)).astype(np.int32))

    def inject_anomaly(self):
        print("Inject anomalies!")
        original_triples = self.triples
        triple_size = len(original_triples)

        if self.isInjectTopK:
            self.num_anomalies = params.num_anomaly_num
            print("###########Inject TOP@K Anomalies##########")
        else:
            self.num_anomalies = int(params.anomaly_ratio * self.num_original_triples)
            params.num_anomaly_num = self.num_anomalies
            print("###########Inject TOP@K% Anomalies##########")

        # idx = random.sample(range(0, self.num_original_triples - 1), num_anomalies)
        idx = random.sample(range(0, self.num_original_triples - 1), self.num_anomalies // 2)
        selected_triples = [original_triples[idx[i]] for i in range(len(idx))]
        anomalies = self.generate_anomalous_triples(selected_triples) + self.generate_anomalous_triples_2(self.num_anomalies // 2)

        triple_label = [(original_triples[i], 0) for i in range(len(original_triples))]
        anomaly_label = [(anomalies[i], 1) for i in range(len(anomalies))]

        triple_anomaly_label = triple_label + anomaly_label
        shuffle(triple_anomaly_label)
        return triple_anomaly_label

#
# dataset = Reader(params.data_dir_FB, "train")
# xxx = dataset.inject_anomaly()
# # print(xxx[0][1])
# # print(xxx[0][0])
# #
# xxx, y, a = dataset.get_data()
# print(xxx[0])
