import time

import torch
import torch.nn as nn
import numpy as np
from utils import *
from network import Flashback
from scipy.sparse import csr_matrix


class FlashbackTrainer():
    """ Instantiates Flashback module with spatial and temporal weight functions.
    Performs loss computation and prediction.
    """

    def __init__(self, lambda_t, lambda_s, lambda_loc, lambda_user, use_weight, transition_graph, friend_graph, use_graph_user):
        """ The hyper parameters to control spatial and temporal decay.
        """
        self.lambda_t = lambda_t
        self.lambda_s = lambda_s

        self.lambda_loc = lambda_loc
        self.lambda_user = lambda_user
        self.use_weight = use_weight
        self.graph = transition_graph
        self.friend_graph = friend_graph
        self.use_graph_user = use_graph_user

    def __str__(self):
        return 'Use flashback training.'

    def parameters(self):
        return self.model.parameters()

    def prepare(self, loc_count, user_count, hidden_size, gru_factory, device):
        f_t = lambda delta_t, user_len: ((torch.cos(delta_t * 2 * np.pi / 86400) + 1) / 2) * torch.exp(
            -(delta_t / 86400 * self.lambda_t))  # hover cosine + exp decay
        f_s = lambda delta_s, user_len: torch.exp(-(delta_s * self.lambda_s))  # exp decay  2个functions
        self.loc_count = loc_count
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.model = Flashback(loc_count, user_count, hidden_size, f_t, f_s, gru_factory, self.lambda_loc,  self.lambda_user, self.use_weight, self.graph, self.friend_graph, self.use_graph_user).to(device)

    def evaluate(self, x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_users):
        """ takes a batch (users x location sequence)
        then does the prediction and returns a list of user x sequence x location
        describing the probabilities for each location at each position in the sequence.
        t, s are temporal and spatial data related to the location sequence x
        y_t, y_s are temporal and spatial data related to the target sequence y.
        Flashback does not access y_t and y_s for prediction!
        """

        self.model.eval()
        out, h = self.model(x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_users)  # (seq_len, user_len, loc_count)

        # seq_len, user_len, loc_count = out.shape
        # out = out.view(-1, self.loc_count)  # (seq_len * batch_size, loc_count)
        # out = out.t()  # (loc_count, seq_len * batch_size)
        # graph = sparse_matrix_to_tensor(self.graph).to(x.device)  # (loc_count, loc_count)
        # graph = graph.t()
        # out = torch.sparse.mm(graph, out)  # (loc_count, seq_len * batch_size)
        # out = out.t()  # (seq_len * batch_size, loc_count)
        # out = torch.reshape(out, (seq_len, user_len, loc_count))

        out_t = out.transpose(0, 1)
        return out_t, h  # model outputs logits

    def loss(self, x, t, t_slot, s, y, y_t, y_t_slot, y_s, h, active_users):
        """ takes a batch (users x location sequence)
        and corresponding targets in order to compute the training loss """

        self.model.train()
        out, h = self.model(x, t, t_slot, s, y_t, y_t_slot, y_s, h,
                            active_users)  # out (seq_len, batch_size, loc_count)
        out = out.view(-1, self.loc_count)  # (seq_len * batch_size, loc_count)

        # out = torch.softmax(out, dim=1)
        # out = torch.log(out + 1e-8)  # 防止log0出现
        # print(out)
        # print(torch.where(torch.isnan(out) == True))

        # values, indices = torch.max(out, dim=1)  # 防止softmax下溢出
        # out = out - values.unsqueeze(1)
        # temp = F.log_softmax(out, dim=1)
        # print(temp)
        #
        # print(torch.where(torch.isnan(temp) == True))
        # print(torch.min(temp, dim=1))
        # print(torch.min(temp))
        # print(torch.max(temp))
        # print(torch.where(torch.isnan(F.log_softmax(out, dim=1)) == True))

        # out = out + 1e-2  # 避免out太小以至于softmax结果为0
        # out = out.t()  # (loc_count, seq_len * batch_size)
        # graph = sparse_matrix_to_tensor(self.graph).to(x.device)  # (loc_count, loc_count)
        # graph = graph.t()
        # out = torch.sparse.mm(graph, out)  # (loc_count, seq_len * batch_size)
        # out = out.t()  # (seq_len * batch_size, loc_count)

        y = y.view(-1)  # (seq_len * batch_size)
        # l = nn.NLLLoss()(out, y)
        l = self.cross_entropy_loss(out, y)
        # print(l.item())
        return l
