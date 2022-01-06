import torch
import torch.nn as nn
from enum import Enum
import time
import numpy as np
from utils import *
import scipy.sparse as sp
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Rnn(Enum):
    """ The available RNN units """

    RNN = 0
    GRU = 1
    LSTM = 2

    @staticmethod
    def from_string(name):
        if name == 'rnn':
            return Rnn.RNN
        if name == 'gru':
            return Rnn.GRU
        if name == 'lstm':
            return Rnn.LSTM
        raise ValueError('{} not supported in --rnn'.format(name))


class RnnFactory():
    """ Creates the desired RNN unit. """

    def __init__(self, rnn_type_str):
        self.rnn_type = Rnn.from_string(rnn_type_str)

    def __str__(self):
        if self.rnn_type == Rnn.RNN:
            return 'Use pytorch RNN implementation.'
        if self.rnn_type == Rnn.GRU:
            return 'Use pytorch GRU implementation.'
        if self.rnn_type == Rnn.LSTM:
            return 'Use pytorch LSTM implementation.'

    def is_lstm(self):
        return self.rnn_type in [Rnn.LSTM]

    def create(self, hidden_size):
        if self.rnn_type == Rnn.RNN:
            return nn.RNN(hidden_size, hidden_size)  # 因为拼接了time embedding
        if self.rnn_type == Rnn.GRU:
            return nn.GRU(hidden_size, hidden_size)
        if self.rnn_type == Rnn.LSTM:
            return nn.LSTM(hidden_size, hidden_size)


class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=device))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=device))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]


class SelfAttn(nn.Module):
    def __init__(self, emb_size, output_size, dropout=0.1):
        super(SelfAttn, self).__init__()
        self.query = nn.Linear(emb_size, output_size, bias=False)
        self.key = nn.Linear(emb_size, output_size, bias=False)
        self.value = nn.Linear(emb_size, output_size, bias=False)

    def forward(self, joint, delta, traj_len):
        delta = torch.sum(delta, -1)  # squeeze the embed dimension
        # joint (N, M, emb), delta (N, M, M, emb), len [N]
        # construct attention mask
        mask = torch.zeros_like(delta, dtype=torch.float32)
        for i in range(mask.shape[0]):
            mask[i, 0:traj_len[i], 0:traj_len[i]] = 1
        # 没有除以根号d
        attn = torch.add(torch.bmm(self.query(joint), self.key(joint).transpose(-1, -2)), delta)  # (N, M, M)
        attn = F.softmax(attn, dim=-1) * mask  # (N, M, M)

        attn_out = torch.bmm(attn, self.value(joint))  # (N, M, emb)

        return attn_out  # (N, M, emb)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20, dropout=0):
        # max_len=5000, dropout=0.1
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (1, seq, dim) -> (seq, 1, dim)
        self.register_buffer('pe', pe)

    def forward(self, x):  # x: (seq, batch, dim)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Flashback(nn.Module):
    """ Flashback RNN: Applies weighted average using spatial and tempoarl data in combination
    of user embeddings to the output of a generic RNN unit (RNN, GRU, LSTM).
    """

    def __init__(self, input_size, user_count, hidden_size, f_t, f_s, rnn_factory, lambda_loc, lambda_user, use_weight,
                 graph, spatial_graph, friend_graph, use_graph_user, use_spatial_graph):
        super().__init__()
        self.input_size = input_size  # POI个数
        self.user_count = user_count
        self.hidden_size = hidden_size
        self.f_t = f_t  # function for computing temporal weight
        self.f_s = f_s  # function for computing spatial weight

        self.lambda_loc = lambda_loc
        self.lambda_user = lambda_user
        self.use_weight = use_weight
        self.use_graph_user = use_graph_user
        self.use_spatial_graph = use_spatial_graph

        self.I = identity(graph.shape[0], format='coo')
        self.graph = sparse_matrix_to_tensor(calculate_random_walk_matrix((graph * self.lambda_loc + self.I).astype(np.float32)))
        # self.interact_graph = sparse_matrix_to_tensor(calculate_random_walk_matrix(interact_graph))

        # self.graph = graph
        self.spatial_graph = spatial_graph
        self.friend_graph = friend_graph

        self.encoder = nn.Embedding(input_size, hidden_size)  # location embedding
        # self.time_encoder = nn.Embedding(24 * 7, hidden_size)  # time embedding
        self.user_encoder = nn.Embedding(user_count, hidden_size)  # user embedding
        self.pref_encoder = nn.Embedding(1, hidden_size * 2)
        self.project_matrix = nn.Linear(hidden_size, hidden_size * 2)  # 将user和location投影到同一片子空间

        # GCN里使用的AXW中的权重矩阵W从这两个随便选一个试试
        # self._gconv_params = LayerParams(self, 'gconv')
        self.loc_gconv_weight = nn.Linear(hidden_size, hidden_size)
        self.user_gconv_weight = nn.Linear(hidden_size, hidden_size)

        self.rnn = rnn_factory.create(hidden_size)  # 改了这里！！！
        # self.pos_encoder = PositionalEncoding(hidden_size)  # seq_len = 20
        # self.attn = nn.MultiheadAttention(hidden_size, 1)
        self.fc = nn.Linear(2 * hidden_size, input_size)  # create outputs in length of locations

    def forward(self, x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_user):
        # 用GCN处理转移graph, 即用顶点i的邻居顶点j来更新i所对应的POI embedding
        seq_len, user_len = x.size()

        graph = self.graph.to(x.device)
        # AX
        loc_emb = self.encoder(torch.LongTensor(list(range(self.input_size))).to(x.device))
        encoder_weight = torch.sparse.mm(graph, loc_emb).to(x.device)  # (input_size, hidden_size)

        if self.use_spatial_graph:
            spatial_graph = (self.spatial_graph * self.lambda_loc + self.I).astype(np.float32)
            spatial_graph = calculate_random_walk_matrix(spatial_graph)
            spatial_graph = sparse_matrix_to_tensor(spatial_graph).to(x.device)  # sparse tensor gpu
            encoder_weight += torch.sparse.mm(spatial_graph, loc_emb).to(x.device)
            encoder_weight /= 2  # 求均值

        if self.use_weight:
            # AXW
            # weights = self._gconv_params.get_weights((self.hidden_size, self.hidden_size))
            # biases = self._gconv_params.get_biases(self.hidden_size, 1.0)
            # encoder_weight = torch.matmul(encoder_weight, weights)  # (input_size, hidden_size)
            # encoder_weight += biases
            # 或者
            encoder_weight = self.loc_gconv_weight(encoder_weight)

        # GLU激活函数
        # encoder_weight = encoder_weight.mul(torch.sigmoid(encoder_weight))
        relu_end = time.time()
        # print('激活: ', relu_end - gcn_end)

        # 是否用GCN来更新user embedding
        if self.use_graph_user:
            I_f = identity(self.friend_graph.shape[0], format='coo')
            friend_graph = (self.friend_graph * self.lambda_user + I_f).astype(np.float32)
            friend_graph = calculate_random_walk_matrix(friend_graph)
            friend_graph = sparse_matrix_to_tensor(friend_graph).to(x.device)
            # AX
            user_emb = self.user_encoder(torch.LongTensor(list(range(self.user_count))).to(x.device))
            user_encoder_weight = torch.sparse.mm(friend_graph, user_emb).to(x.device)  # (user_count, hidden_size)

            if self.use_weight:
                user_encoder_weight = self.user_gconv_weight(user_encoder_weight)
            p_u = torch.index_select(user_encoder_weight, 0, active_user.squeeze())
        else:
            p_u = self.user_encoder(active_user)  # (1, user_len, hidden_size)
            p_u = p_u.view(user_len, self.hidden_size)  # (user_len, hidden_size)

        new_x_emb = []
        for i in range(seq_len):
            new_x_emb.append(torch.index_select(encoder_weight, 0, x[i]))

        x_emb = torch.stack(new_x_emb, dim=0)

        # x_time_emb = self.time_encoder(t_slot)
        # x_emb = torch.cat([x_emb, x_time_emb], dim=-1)  # (seq_len, user_len, hidden_size * 2)
        # out, h = self.rnn(new_x_emb, h)  # (seq_len, user_len, hidden_size)

        # x_pos_emb = self.pos_encoder(x_emb)  # 加位置编码
        # mask = ~torch.tril(torch.ones([seq_len, seq_len])).bool().to(x.device)  # mask 矩阵
        # out, _ = self.attn(x_pos_emb, x_pos_emb, x_pos_emb, attn_mask=mask)
        out, h = self.rnn(x_emb, h)  # (seq_len, user_len, hidden_size)
        out_w = torch.zeros(seq_len, user_len, self.hidden_size, device=x.device)

        p_proj_u = torch.tanh(self.project_matrix(p_u))  # (user_len, hidden_size * 2)
        x_proj_emb = torch.tanh(self.project_matrix(x_emb)).permute(1, 0, 2)  # (user_len, seq_len, hidden_size * 2)
        preference_emb = self.pref_encoder(torch.LongTensor([0]).to(x.device))
        # 计算用户对check-in中的每个POI的偏好以及偏好的类型  (user_len, seq_len)  让这部分在CPU上计算
        # user_loc_similarity = calculate_preference_similarity(p_proj_u.cpu(), x_proj_emb.cpu(),
        #                                                       preference_emb.cpu()).to(x.device)
        user_loc_similarity = compute_preference(p_proj_u.cpu(), x_proj_emb.cpu(),
                                                 preference_emb.cpu()).to(x.device)
        user_loc_time = time.time()
        # print('计算用户相似性: ', user_loc_time - new_x_time)
        for i in range(seq_len):
            sum_w = torch.zeros(user_len, 1, device=x.device)  # (200, 1)
            for j in range(i + 1):
                dist_t = t[i] - t[j]
                dist_s = torch.norm(s[i] - s[j], dim=-1)
                a_j = self.f_t(dist_t, user_len)  # (user_len, )
                b_j = self.f_s(dist_s, user_len)
                a_j = a_j.unsqueeze(1)  # (user_len, 1)
                b_j = b_j.unsqueeze(1)
                w_j = a_j * b_j + 1e-10  # small epsilon to avoid 0 division
                w_j = w_j * user_loc_similarity[:, j].unsqueeze(1)  # (user_len, 1)
                sum_w += w_j
                out_w[i] += w_j * out[j]  # (user_len, hidden_size)
            out_w[i] /= sum_w

        # 计算user所喜好的POI
        # interact = self.interact_graph.to(x.device)  # (user_count, input_size)
        # user_interact = torch.sparse.mm(interact, encoder_weight)  # (user_count, hidden_size)
        # user_p = torch.index_select(user_interact, 0, active_user.squeeze())  # (user_len, hidden_size)
        out_pu = torch.zeros(seq_len, user_len, 2 * self.hidden_size, device=x.device)
        for i in range(seq_len):
            out_pu[i] = torch.cat([out_w[i], p_u], dim=1)  # (user_len, hidden_size * 2)

        y_linear = self.fc(out_pu)  # (seq_len, user_len, loc_count)

        return y_linear, h


'''
~~~ h_0 strategies ~~~
Initialize RNNs hidden states
'''


def create_h0_strategy(hidden_size, is_lstm):
    if is_lstm:
        return LstmStrategy(hidden_size, FixNoiseStrategy(hidden_size), FixNoiseStrategy(hidden_size))
    else:
        return FixNoiseStrategy(hidden_size)


class H0Strategy():

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def on_init(self, user_len, device):
        pass

    def on_reset(self, user):
        pass

    def on_reset_test(self, user, device):
        return self.on_reset(user)


class FixNoiseStrategy(H0Strategy):
    """ use fixed normal noise as initialization """

    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        mu = 0
        sd = 1 / self.hidden_size
        self.h0 = torch.randn(self.hidden_size, requires_grad=False) * sd + mu

    def on_init(self, user_len, device):
        hs = []
        for i in range(user_len):
            hs.append(self.h0)
        return torch.stack(hs, dim=0).view(1, user_len, self.hidden_size).to(device)  # (1, 200, 10)

    def on_reset(self, user):
        return self.h0


class LstmStrategy(H0Strategy):
    """ creates h0 and c0 using the inner strategy """

    def __init__(self, hidden_size, h_strategy, c_strategy):
        super(LstmStrategy, self).__init__(hidden_size)
        self.h_strategy = h_strategy
        self.c_strategy = c_strategy

    def on_init(self, user_len, device):
        h = self.h_strategy.on_init(user_len, device)
        c = self.c_strategy.on_init(user_len, device)
        return h, c

    def on_reset(self, user):
        h = self.h_strategy.on_reset(user)
        c = self.c_strategy.on_reset(user)
        return h, c
