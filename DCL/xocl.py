# -*- coding: utf-8 -*-
# @Time   : 2020/8/31
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

# UPDATE:
# @Time   : 2020/9/16, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com

r"""
LightGCN
################################################

Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN
"""
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class XOCL(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(XOCL, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        

        # load parameters info
        self.latent_dim = config[
            "embedding_size"
        ]  # int type:the embedding size of lightGCN
        self.n_layers = config["n_layers"]  # int type:the layer num of lightGCN
        self.reg_weight = config[
            "reg_weight"
        ]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]
        self.tau = config['tau']
        self.eps = config['eps']
        self.cl_rate = config['cl_rate']
        self.cl_layer = config['cl_layer']
        self.temp = config['temp']
        self.neg_num = config['train_neg_sample_args']['sample_num']

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device).coalesce()
        self.uiA_dict = self.get_uiA_dict(dataset['user_id'], dataset['item_id'])
        self.iu_dict = self.get_iu_dict(dataset['user_id'], dataset['item_id'])
        self.pnitems_dict = self.get_pos_neg_items_dict(dataset['user_id'], dataset['item_id'])
        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_uiA_dict(self, user_inter, item_inter):
        node0 = user_inter.tolist()
        node1 = item_inter.tolist()
        uiA_dict = dict()
        for idx in range(len(node0)):
            if node0[idx] not in uiA_dict:
                neighbors = node1[idx]
                uiA_dict.update({node0[idx] : {neighbors}})
            else:
                uiA_dict[node0[idx]].add(node1[idx])
        return uiA_dict
    
    def get_iu_dict(self, user_inter, item_inter):
        node0 = user_inter.tolist()
        node1 = item_inter.tolist()
        iu_dict = dict()
        for idx in range(len(node1)):
            if node1[idx] not in iu_dict:
                neighbors = node0[idx]
                iu_dict.update({node1[idx] : {neighbors}})
            else:
                iu_dict[node1[idx]].add(node0[idx])
        return iu_dict

    def get_pos_neg_items_dict(self, user_inter, item_inter):
        pnItems_dict = dict()

        inter_user_id = user_inter.numpy()
        inter_item_id = item_inter.numpy()

        neg_ids = []
        for user in inter_user_id:
            key_neg_ids = []
            while len(key_neg_ids) < self.neg_num:
                randnum = np.random.randint(1, self.n_items, self.neg_num)
                key_neg_ids.extend([
                    neg_id 
                    for neg_id in randnum 
                    if neg_id not in self.uiA_dict[user]
                ])
            if len(key_neg_ids) > self.neg_num:
                key_neg_ids = key_neg_ids[:self.neg_num]
            neg_ids.append(key_neg_ids)

        neg_uds = []
        for item in inter_item_id:
            key_neg_uds = []
            while len(key_neg_uds) < self.neg_num:
                randnum = np.random.randint(1, self.n_users, self.neg_num)
                key_neg_uds.extend([
                    neg_ud 
                    for neg_ud in randnum 
                    if neg_ud not in self.iu_dict[item]
                ])
            if len(key_neg_uds) > self.neg_num:
                key_neg_uds = key_neg_uds[:self.neg_num]
            neg_uds.append(key_neg_uds)

        neg_item_id = np.array(neg_ids)
        neg_user_id = np.array(neg_uds)

        pnItems_dict.update({'user_id':inter_user_id})
        pnItems_dict.update({'pos_item_id':inter_item_id})
        pnItems_dict.update({'neg_item_id':neg_item_id})
        pnItems_dict.update({'neg_user_id':neg_user_id})

        return pnItems_dict
    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, perturbed=False):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = []

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            if perturbed:
                random_noise = torch.rand_like(all_embeddings).to(self.device)
                all_embeddings = all_embeddings + torch.sign(all_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            embeddings_list.append(all_embeddings)
        
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(embeddings_list[self.cl_layer], [self.n_users, self.n_items])
        if perturbed:
            return user_all_embeddings, item_all_embeddings,user_all_embeddings_cl, item_all_embeddings_cl
        return user_all_embeddings, item_all_embeddings
    
    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)
    
    def cal_cl_loss(self, idx, user_view1,user_view2,item_view1,item_view2):
        u_idx = torch.unique(idx[0].type(torch.long)).to(self.device)
        i_idx = torch.unique(idx[1].type(torch.long)).to(self.device)
        user_row_loss = self.InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
        item_row_loss = self.InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)
        row_loss = user_row_loss + item_row_loss
        return row_loss
    
    def cal_interCL(self, u, pi, ni, nu, u_emb, i_emb):
        user_d3 = torch.unsqueeze(u, 1)
        pos_item_d3 = torch.unsqueeze(pi, 1)
        neg_item = ni
        neg_user = nu

        u_norm = u_emb[user_d3]
        pos_norm = i_emb[pos_item_d3]
        neg_norm = i_emb[neg_item]
        neg_u_norm = u_emb[neg_user]
        pos_score1 = torch.exp(torch.mul(u_norm, pos_norm).sum(-1) / self.tau).squeeze(dim = -1)
        neg_score1 = torch.exp(torch.mul(u_norm, neg_norm).sum(-1) / self.tau).sum(dim = -1)
        interCL1 = torch.mean(-torch.log(1e-10 + pos_score1 / (pos_score1 + neg_score1)))

        neg_score2 = torch.exp(torch.mul(pos_norm, neg_u_norm).sum(-1) / self.tau).sum(dim = -1)
        interCL2 = torch.mean(-torch.log(1e-10 + pos_score1 / (pos_score1 + neg_score2)))
        interCL = interCL1 + interCL2
        return interCL
    
    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        edge_index = interaction.cpu().numpy()

        user = torch.tensor(self.pnitems_dict['user_id'][edge_index]).to(self.device)
        pos_item = torch.tensor(self.pnitems_dict['pos_item_id'][edge_index]).to(self.device)
        neg_item = torch.tensor(self.pnitems_dict['neg_item_id'][edge_index]).to(self.device)
        neg_user = torch.tensor(self.pnitems_dict['neg_user_id'][edge_index]).to(self.device)

        user_all_embeddings, item_all_embeddings, cl_user_emb, cl_item_emb= self.forward(perturbed=True)

        inter_cl_loss = self.cal_interCL(user, pos_item, neg_item, neg_user, user_all_embeddings, item_all_embeddings)
        cl_loss = self.cal_cl_loss([user, pos_item],user_all_embeddings,cl_user_emb,item_all_embeddings,cl_item_emb)
        loss = inter_cl_loss + self.cl_rate * cl_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
