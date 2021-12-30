import torch
import torch.nn as nn
from enum import Enum
import time
from utils import *
import scipy.sparse as sp

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


class Flashback(nn.Module):
    """ Flashback RNN: Applies weighted average using spatial and tempoarl data in combination
    of user embeddings to the output of a generic RNN unit (RNN, GRU, LSTM).
    """

    def __init__(self, input_size, user_count, hidden_size, f_t, f_s, rnn_factory, graph, friend_graph):
        super().__init__()
        self.input_size = input_size  # POI个数
        self.user_count = user_count
        self.hidden_size = hidden_size
        self.f_t = f_t  # function for computing temporal weight
        self.f_s = f_s  # function for computing spatial weight
        self.graph = graph
        self.friend_graph = friend_graph

        self.encoder = nn.Embedding(input_size, hidden_size)  # location embedding
        self.temp_encoder = nn.Embedding(input_size, hidden_size)
        # self.time_encoder = nn.Embedding(24 * 7, hidden_size)  # time embedding
        self.user_encoder = nn.Embedding(user_count, hidden_size)  # user embedding
        self.pref_encoder = nn.Embedding(1, hidden_size * 2)
        self.project_matrix = nn.Linear(hidden_size, hidden_size * 2)  # 将user和location投影到同一片子空间

        # 学习单位矩阵的自适应系数 r * I
        # self.identity_weight = torch.FloatTensor(input_size, 1)
        # nn.init.uniform_(self.identity_weight)

        # GCN里使用的AXW中的权重矩阵W从这两个随便选一个试试
        # self._gconv_params = LayerParams(self, 'gconv')
        # self.gconv_weight = nn.Linear(hidden_size, hidden_size)

        self.rnn = rnn_factory.create(hidden_size)  # 改了这里！！！
        self.fc = nn.Linear(2 * hidden_size, input_size)  # create outputs in length of locations

    def forward(self, x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_user):
        # 用GCN处理转移graph, 即用顶点i的邻居顶点j来更新i所对应的POI embedding
        # graph = coo_matrix(self.graph)
        # I = coo_matrix(identity(graph.shape[0]))

        I = identity(self.graph.shape[0], format='coo')
        # I_f = identity(self.friend_graph.shape[0], format='coo')
        # graph = sparse_matrix_to_tensor(self.graph)  # (loc_count, loc_count)
        # I = sparse_matrix_to_tensor(identity(graph.shape[0]))  # sparse tensor

        # 构造自适应的单位矩阵
        # I = torch.sparse.mm(I, self.identity_weight.cpu())  # dense tensor
        # I = sp.diags(np.array(I).flatten())  # sparse dia_matrix 稀疏对角矩阵
        # I = sparse_matrix_to_tensor(I).to(x.device)  # sparse tensor

        graph = (self.graph + I)  # A + r * I  # sparse matrix
        # friend_graph = (self.friend_graph + I_f)
        # graph_t = calculate_reverse_random_walk_matrix(graph)
        graph = calculate_random_walk_matrix(graph)
        # friend_graph = calculate_random_walk_matrix(friend_graph)

        graph = sparse_matrix_to_tensor(graph).to(x.device)  # sparse tensor gpu
        # friend_graph = sparse_matrix_to_tensor(friend_graph).to(x.device)
        # graph_t = sparse_matrix_to_tensor(graph_t)

        # AX
        loc_emb = self.encoder(torch.LongTensor(list(range(self.input_size))).to(x.device))
        encoder_weight = torch.sparse.mm(graph, loc_emb).to(x.device)  # (input_size, hidden_size)

        # user_emb = self.user_encoder(torch.LongTensor(list(range(self.user_count))).to(x.device))
        # user_encoder_weight = torch.sparse.mm(friend_graph, user_emb).to(x.device)  # (user_count, hidden_size)
        # encoder_weight = (encoder_weight + torch.sparse.mm(graph_t, loc_emb).to(x.device)) / 2  # (A X + A^-1 X) / 2

        # AXW
        # weights = self._gconv_params.get_weights((self.hidden_size, self.hidden_size))
        # biases = self._gconv_params.get_biases(self.hidden_size, 1.0)
        # encoder_weight = torch.matmul(encoder_weight, weights)  # (input_size, hidden_size)
        # encoder_weight += biases
        # 或者
        # encoder_weight = self.gconv_weight(encoder_weight)

        # self.encoder.weight = nn.Parameter(encoder_weight)
        # self.temp_encoder.weight = nn.Parameter(encoder_weight)  # !!!!!!!

        # temp_loc_encoder = nn.Embedding.from_pretrained(encoder_weight)  # 临时的location embedding
        seq_len, user_len = x.size()

        new_x_emb = []
        for i in range(seq_len):
            new_x_emb.append(torch.index_select(encoder_weight, 0, x[i]))

        x_emb = torch.stack(new_x_emb, dim=0)

        # x_emb = temp_loc_encoder(x)
        # x_emb = self.encoder(x)  # (seq_len, user_len, hidden_size)
        # x_emb = self.temp_encoder(x)  # (seq_len, user_len, hidden_size)

        # x_time_emb = self.time_encoder(t_slot)
        # new_x_emb = torch.cat([x_emb, x_time_emb], dim=-1)  # (seq_len, user_len, hidden_size * 2)
        # out, h = self.rnn(new_x_emb, h)  # (seq_len, user_len, hidden_size)

        out, h = self.rnn(x_emb, h)  # (seq_len, user_len, hidden_size)
        # print('h', h.size())  # (1, user_len, hidden_size)
        out_w = torch.zeros(seq_len, user_len, self.hidden_size, device=x.device)

        # p_u = torch.index_select(user_encoder_weight, 0, active_user.squeeze())
        p_u = self.user_encoder(active_user)  # (1, user_len, hidden_size)
        p_u = p_u.view(user_len, self.hidden_size)  # (user_len, hidden_size)

        p_proj_u = torch.tanh(self.project_matrix(p_u))  # (user_len, hidden_size * 2)
        x_proj_emb = torch.tanh(self.project_matrix(x_emb)).permute(1, 0, 2)  # (user_len, seq_len, hidden_size * 2)
        preference_emb = self.pref_encoder(torch.LongTensor([0]).to(x.device))
        # 计算用户对check-in中的每个POI的偏好以及偏好的类型  (user_len, seq_len)  让这部分在CPU上计算
        user_loc_similarity = calculate_preference_similarity(p_proj_u.cpu(), x_proj_emb.cpu(),
                                                              preference_emb.cpu()).to(x.device)

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

        out_pu = torch.zeros(seq_len, user_len, 2 * self.hidden_size, device=x.device)
        for i in range(seq_len):
            out_pu[i] = torch.cat([out_w[i], p_u], dim=1)  # (user_len, hidden_size * 2)

        y_linear = self.fc(out_pu)
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
