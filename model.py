import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import params
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphAttentionLayer1(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=False):
        super(GraphAttentionLayer1, self).__init__()
        self.in_features = in_features  
        self.out_features = out_features 
        self.dropout = dropout 
        self.alpha = alpha 
        self.concat = concat 

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414) 
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  

        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp):
        """
        inp: input_fea [Batch_size, N, in_features]
        """
        h = torch.matmul(inp, self.W)  # [batch_size, N, out_features]
        N = h.size()[1]  
        B = h.size()[0]  # B batch_size

        a = h[:, 0, :].unsqueeze(1).repeat(1, N, 1)  # [batch_size, N, out_features]
        a_input = torch.cat((h, a), dim=2)  # [batch_size, N, 2*out_features]

        # a_input = torch.cat([h.repeat(1, 1, N).view(params.batch_size, N * N, -1), h.repeat(1, N, 1)], dim=2).view(params.batch_size, N, -1, 2 * self.out_features)

        e = self.leakyrelu(torch.matmul(a_input, self.a))
        # [batch_size, N, 1] 

        attention = F.softmax(e, dim=1)  # [batch_size, N, 1]
        attention = attention - 0.001
        attention = (attention + abs(attention)) / 2.0
        # print(attention)
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout
        # print(attention)
        attention = attention.view(B, 1, N)
        h_prime = torch.matmul(attention, h).squeeze(1)  # [batch_size, 1, N]*[batch_size, N, out_features] => [batch_size, 1, out_features]
       
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class BiLSTM_Attention(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, alpha):
        super(BiLSTM_Attention, self).__init__()
        # self.ent_embeddings = nn.Embedding(params.total_ent + 1, params.embedding_dim)
        # self.rel_embeddings = nn.Embedding(params.total_rel + 1, params.embedding_dim)
        # self.init_weights()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = 3
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        #self.fc = nn.Linear(hidden_size * 2 * self.seq_length, num_classes)  # 2 for bidirection

        self.attention = GraphAttentionLayer1(self.hidden_size * 2 * self.seq_length, self.hidden_size * 2 * self.seq_length, dropout=dropout, alpha=alpha, concat=False)
        # self.attentions = [GraphAttentionLayer(self.hidden_size * 2 * self.seq_length, self.hidden_size * 2 * self.seq_length, dropout=dropout, alpha=alpha, concat=False) for _ in
        #                    range(nheads)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)
        self.ent_embeddings = nn.Embedding(params.total_ent, params.embedding_dim)
        self.rel_embeddings = nn.Embedding(params.total_rel, params.embedding_dim)

        # print(toarray_float(ent_vec).shape)
        # print(params.total_ent, params.total_rel, params.embedding_dim)
        # self.ent_embeddings.weight.data.copy_(torch.from_numpy(ent_vec))
        # self.rel_embeddings.weight.data.copy_(torch.from_numpy(rel_vec))
        uniform_range = 6 / np.sqrt(params.embedding_dim)
        self.ent_embeddings.weight.data.uniform_(-uniform_range, uniform_range)
        self.rel_embeddings.weight.data.uniform_(-uniform_range, uniform_range)

    def forward(self, batch_h, batch_r, batch_t):
        # head, relation, tail = torch.chunk(inputTriple,
        #                                    chunks=3,
        #                                    dim=1)
        # head = torch.squeeze(self.ent_embeddings(head), dim=1)
        # tail = torch.squeeze(self.ent_embeddings(tail), dim=1)
        # relation = torch.squeeze(self.rel_embeddings(relation), dim=1)
        # print(batch_t.cpu())
        # print(batch_r.cpu())
        head = self.ent_embeddings(batch_h)
        relation = self.rel_embeddings(batch_r)
        tail = self.ent_embeddings(batch_t)

        batch_triples_emb = torch.cat((head, relation), dim=1)
        batch_triples_emb = torch.cat((batch_triples_emb, tail), dim=1)
        x = batch_triples_emb.view(-1, 3, params.BiLSTM_input_size)
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
        # batch_triples_emb = batch_triples_emb.view(-1, 3, params.BiLSTM_input_size)
        # # print('input_batch.shape:', batch_triples_emb.shape)
        # batch_size = x.size(0)
        # [B, 3, input_size] B = batch_size * 2 * 2 * (num_neighbor+1)
        # x = x.to(device)

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(params.device)# 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(params.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (B, seq_length, hidden_size*2)

        # print('out_lstm', out_lstm.shape)
        out = out.reshape(-1, self.hidden_size * 2 * self.seq_length)
        out = out.reshape(-1, params.num_neighbor + 1, self.hidden_size * 2 * self.seq_length)
        # [batch_size * 2 * 2, num_neighbor+1, dim_embedding] dim_embedding = hidden_size * 2 * seq_length

        out_att = self.attention(out)
        # [batch_size * 2 * 2, dim_embedding]
        # out_att = self.attention_0(out[0:params.num_neighbor + 1])
        # print('input to linear', out.shape)
        # Decode the hidden state of the last time step
        #out = self.fc(out_lstm)
        out = out.reshape(-1, params.num_neighbor * 2 + 2, self.hidden_size * 2 * self.seq_length)
        return out[:, 0, :], out_att
# class BiLSTM_Attention(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, dropout, alpha):
#         super(BiLSTM_Attention, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.seq_length = 3
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
#         #self.fc = nn.Linear(hidden_size * 2 * self.seq_length, num_classes)  # 2 for bidirection
#
#         self.attention = GraphAttentionLayer1(self.hidden_size * 2 * self.seq_length, self.hidden_size * 2 * self.seq_length, dropout=dropout, alpha=alpha, concat=False)
#         # self.attentions = [GraphAttentionLayer(self.hidden_size * 2 * self.seq_length, self.hidden_size * 2 * self.seq_length, dropout=dropout, alpha=alpha, concat=False) for _ in
#         #                    range(nheads)]
#         # for i, attention in enumerate(self.attentions):
#         #     self.add_module('attention_{}'.format(i), attention)
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         # [B, 3, input_size] B = batch_size * 2 * 2 * (num_neighbor+1)
#         x = x.to(device)
#         h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 2 for bidirection
#         c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
#
#         # Forward propagate LSTM
#         out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (B, seq_length, hidden_size*2)
#
#         # print('out_lstm', out_lstm.shape)
#         out = out.reshape(-1, self.hidden_size * 2 * self.seq_length)
#         out = out.reshape(-1, params.num_neighbor + 1, self.hidden_size * 2 * self.seq_length)
#         # [batch_size * 2 * 2, num_neighbor+1, dim_embedding] dim_embedding = hidden_size * 2 * seq_length
#
#         out_att = self.attention(out)
#         # [batch_size * 2 * 2, dim_embedding]
#         # out_att = self.attention_0(out[0:params.num_neighbor + 1])
#         # print('input to linear', out.shape)
#         # Decode the hidden state of the last time step
#         #out = self.fc(out_lstm)
#         out = out.reshape(-1, params.num_neighbor * 2 + 2, self.hidden_size * 2 * self.seq_length)
#         return out[:, 0, :], out_att

class TransE(nn.Module):
    def __init__(self):
        super(TransE, self).__init__()
        self.params = params
        self.ent_embeddings = nn.Embedding(params.total_ent, params.embedding_dim)
        self.rel_embeddings = nn.Embedding(params.total_rel, params.embedding_dim)

        self.criterion = nn.MarginRankingLoss(params.margin)

        self.init_weights()

        logging.info('Initialized the model successfully!')

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def get_score(self, h, t, r):
        return torch.norm(h + r - t, self.params.p_norm, -1)

    def forward(self, batch_h, batch_t, batch_r, batch_y):
        # triples = batch_triples.reshape(-1, 2, 3)
        # batch_h = Variable(torch.LongTensor(batch_h).cuda())
        # batch_t = torch.LongTensor(batch_t).cuda()
        # batch_r = torch.LongTensor(batch_r).cuda()
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)

        score = self.get_score(h, t, r)

        pos_score = score[0: int(len(score) / 2)]
        neg_score = score[int(len(score) / 2): len(score)]
        loss = self.criterion(pos_score, neg_score, torch.Tensor([-1]).to(params.device))
        # loss = self.criterion(pos_score.cpu(), neg_score.cpu(), torch.Tensor([-1]))
        return loss, pos_score, neg_score


class DistMult(nn.Module):
    def __init__(self):
        super(DistMult, self).__init__()
        self.ent_embeddings = nn.Embedding(params.total_ent, params.embedding_dim)
        self.rel_embeddings = nn.Embedding(params.total_rel, params.embedding_dim)

        # self.criterion = nn.Softplus()
        self.criterion = nn.MarginRankingLoss(params.margin)

        self.init_weights()

        logging.info('Initialized the model successfully!')

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def get_score(self, h, t, r):
        return -torch.norm(h * t * r, params.p_norm, -1)
        # return torch.norm(h * t * r, params.p_norm, dim=1)

    def forward(self, batch_h, batch_t, batch_r, batch_y):
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        y = torch.from_numpy(batch_y).type(torch.FloatTensor)

        score = self.get_score(h, t, r)
        # score = torch.sigmoid(score)
        # score = torch.norm(score, params.p_norm, -1)
        pos_score = score[0: int(len(score) / 2)]
        neg_score = score[int(len(score) / 2): len(score)]

        # regul = torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)
        # loss = torch.mean(self.criterion(score * y)) + params.lmbda * regul
        # loss = self.criterion(pos_score.cpu(), neg_score.cpu(), torch.Tensor([-1]))
        loss = self.criterion(pos_score, neg_score, torch.Tensor([-1]).to(params.device))

        return loss, pos_score, neg_score


class ComplEx(nn.Module):
    def __init__(self):
        super(ComplEx, self).__init__()

        self.ent_re_embeddings = nn.Embedding(
            params.total_ent, params.embedding_dim
        )
        self.ent_im_embeddings = nn.Embedding(
            params.total_ent, params.embedding_dim
        )
        self.rel_re_embeddings = nn.Embedding(
            params.total_rel, params.embedding_dim
        )
        self.rel_im_embeddings = nn.Embedding(
            params.total_rel, params.embedding_dim
        )
        # self.criterion = nn.Softplus()
        self.criterion = nn.MarginRankingLoss(params.margin)
        self.init_weights()

        logging.info('Initialized the model successfully!')

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)

    def get_score(self, h_re, h_im, t_re, t_im, r_re, r_im):
        return -torch.norm(
            h_re * t_re * r_re
            + h_im * t_im * r_re
            + h_re * t_im * r_im
            - h_im * t_re * r_im,
            params.p_norm,
            -1
        )

    def forward(self, batch_h, batch_t, batch_r, batch_y):
        h_re = self.ent_re_embeddings(batch_h)
        h_im = self.ent_im_embeddings(batch_h)
        t_re = self.ent_re_embeddings(batch_t)
        t_im = self.ent_im_embeddings(batch_t)
        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)

        y = torch.from_numpy(batch_y).type(torch.FloatTensor)

        score = self.get_score(h_re, h_im, t_re, t_im, r_re, r_im)

        pos_score = score[0: int(len(score) / 2)]
        neg_score = score[int(len(score) / 2): len(score)]

        regul = (
            torch.mean(h_re ** 2)
            + torch.mean(h_im ** 2)
            + torch.mean(t_re ** 2)
            + torch.mean(t_im ** 2)
            + torch.mean(r_re ** 2)
            + torch.mean(r_im ** 2)
        )
        # loss = torch.mean(self.criterion(score * y)) + params.lmbda * regul
        loss = self.criterion(pos_score, neg_score, torch.Tensor([-1]).to(params.device))
        return loss, pos_score, neg_score