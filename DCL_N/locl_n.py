# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import numpy as np

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from logging import getLogger

class LOCL_N(GeneralRecommender):

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LOCL_N, self).__init__(config, dataset)

        # load dataset info
        self.NODE_ID = 'node_id'
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.cl_layer = config['cl_layer']
        self.tau = config['tau']
        self.reg_weight = config['reg_weight']
        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device).coalesce()
        # self.interaction_matrix_torch = self.get_interaction_matrix_torch(dataset)
        self.uiA_dict = self.get_uiA_dict()

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']
        
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
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        self.diag = torch.from_numpy(diag).to(self.device)
        D = sp.diags(diag)
        L = D @ A @ D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL
    
    def get_uiA_dict(self):
        node0 = self.norm_adj_matrix.indices()[0].cpu().tolist()
        node1 = self.norm_adj_matrix.indices()[1].cpu().tolist()
        values = self.norm_adj_matrix.values().cpu().tolist()
        uiA_dict = dict()
        for idx in range(len(node0)):
            if node0[idx] not in uiA_dict:
                neighbors = node1[idx]
                neighbors_value = values[idx]
                uiA_dict.update({node0[idx] : {'neighbors': [neighbors], 'neighbors_value': [neighbors_value]}})
            else:
                uiA_dict[node0[idx]]['neighbors'].append(node1[idx])
                uiA_dict[node0[idx]]['neighbors_value'].append(values[idx])
        return uiA_dict

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)

        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings
    
    def cal_user_neicl(self, user, user_all_embeddings, item_all_embeddings):        
        u_norm = F.normalize(user_all_embeddings[user], dim = 1)
        i_norm = F.normalize(item_all_embeddings, dim = 1)
        
        scores = torch.matmul(u_norm, i_norm.transpose(0, 1))
        u_list = []
        i_list = []
        weight_list = []
        uidx_list = []
        sparse_shape = (len(user), self.n_items)
        i = 0
        for e in user:
            u_list.extend([i] * len(self.uiA_dict[e]['neighbors']))
            i_list.extend(self.uiA_dict[e]['neighbors'])
            weight_list.extend(self.uiA_dict[e]['neighbors_value'])
            uidx_list.extend([e] * len(self.uiA_dict[e]['neighbors']))
            i += 1
        weight = torch.tensor(weight_list).to(self.device)
        u_array = np.array(u_list)
        i_array = np.array(i_list) - self.n_users
        pos_indices = torch.tensor(np.vstack((u_array, i_array)))

        pos_scores_tau1 = torch.exp(scores[[u_array, i_array]] / self.tau)
        pos_sparse_tau1 = torch.sparse_coo_tensor(pos_indices.to(self.device), pos_scores_tau1, size = sparse_shape)
        pos_sum_tau1 = torch.sparse.sum(pos_sparse_tau1, dim = 1).to_dense()

        pos_scores = torch.exp(scores[[u_array, i_array]] / self.tau)
        # pos_scores = pos_scores * weight
        pos_sparse = torch.sparse_coo_tensor(pos_indices.to(self.device), pos_scores, size = sparse_shape)
        pos_sum = torch.sparse.sum(pos_sparse, dim = 1).to_dense()

        all_sum = torch.sum(torch.exp(scores / self.tau), dim = 1)
        neg_sum = (all_sum - pos_sum)
        gama = 1e-10 
        L = -torch.mean(torch.log(gama + pos_sum_tau1 / neg_sum))
        return L
    
    def cal_item_neicl(self, item, user_all_embeddings, item_all_embeddings):        
        u_norm = F.normalize(user_all_embeddings, dim = 1)
        i_norm = F.normalize(item_all_embeddings[item], dim = 1)
        scores = torch.matmul(i_norm, u_norm.transpose(0, 1))
        item_gID = item + self.n_users
        u_list = []
        i_list = []
        weight_list = []
        iidx_list = []
        sparse_shape = (len(item), self.n_users)
        idx = 0
        for e in item_gID:
            i_list.extend([idx] * len(self.uiA_dict[e]['neighbors']))
            u_list.extend(self.uiA_dict[e]['neighbors'])
            weight_list.extend(self.uiA_dict[e]['neighbors_value'])
            iidx_list.extend([e] * len(self.uiA_dict[e]['neighbors']))
            idx += 1
        weight = torch.tensor(weight_list).to(self.device)
        i_array = np.array(i_list)
        u_array = np.array(u_list)
        pos_indices = torch.tensor(np.vstack((i_array, u_array)))

        pos_scores_tau1 = torch.exp(scores[[i_array, u_array]] / self.tau)
        pos_sparse_tau1 = torch.sparse_coo_tensor(pos_indices.to(self.device), pos_scores_tau1, size = sparse_shape)
        pos_sum_tau1 = torch.sparse.sum(pos_sparse_tau1, dim = 1).to_dense()

        pos_scores = torch.exp(scores[[i_array, u_array]] / self.tau)
        pos_sparse = torch.sparse_coo_tensor(pos_indices.to(self.device), pos_scores, size = sparse_shape)
        pos_sum = torch.sparse.sum(pos_sparse, dim = 1).to_dense()

        all_sum = torch.sum(torch.exp(scores / self.tau), dim = 1)
        neg_sum = (all_sum - pos_sum)
        gama = 1e-10 
        L = -torch.mean(torch.log(gama + pos_sum_tau1 / neg_sum))
        return L

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        node = interaction[self.NODE_ID]
        user, item = self.get_UIID(node)

        user_all_embeddings, item_all_embeddings = self.forward()
        loss_list = []

        if len(user) > 0:
            user_neicl_loss = self.cal_user_neicl(user, user_all_embeddings, item_all_embeddings)
            loss_list.append(user_neicl_loss)
        if len(item) > 0:
            item_neicl_loss = self.cal_item_neicl(item, user_all_embeddings, item_all_embeddings)
            loss_list.append(item_neicl_loss)

        final_loss = 0
        for loss in loss_list:
            final_loss = final_loss + loss
        return final_loss

    def get_UIID(self, node):
            node_list = node.cpu().tolist()
            user_list = []
            item_list = []
            for node in node_list:
                if node < self.n_users:
                    user_list.append(node)
                else:
                    item_list.append(node)
            user_array = np.array(user_list)
            item_array = np.array(item_list) - self.n_users
            return user_array, item_array

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
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)