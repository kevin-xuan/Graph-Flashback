import torch
import torch.nn as nn
from enum import Enum
from utils import load_graph_data


def calculate_similarity(m1, m2, pref, device):
    """
        m1: (user_len, hidden_size)
        m2：(user_len, seq_len, hidden_size)
    """
    user_len = m1.shape[0]
    seq_len = m2.shape[1]
    pref = pref.squeeze()
    similarity = torch.zeros(user_len, seq_len, dtype=torch.float32).to(device)
    for i in range(user_len):
        v1 = m1[i]
        for j in range(seq_len):
            v2 = m2[i][j]
            similarity[i][j] = (1 + torch.cosine_similarity(v1 + pref, v2, dim=0).item()) / 2  # 归一化到[0, 1]

    return similarity


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
            return nn.RNN(hidden_size, hidden_size)
        if self.rnn_type == Rnn.GRU:
            return nn.GRU(hidden_size, hidden_size)
        if self.rnn_type == Rnn.LSTM:
            return nn.LSTM(hidden_size, hidden_size)


class Flashback(nn.Module):
    """ Flashback RNN: Applies weighted average using spatial and tempoarl data in combination
    of user embeddings to the output of a generic RNN unit (RNN, GRU, LSTM).
    """

    def __init__(self, input_size, user_count, hidden_size, f_t, f_s, rnn_factory):
        super().__init__()
        self.input_size = input_size  # POI个数
        self.user_count = user_count
        self.hidden_size = hidden_size
        self.f_t = f_t  # function for computing temporal weight
        self.f_s = f_s  # function for computing spatial weight

        self.encoder = nn.Embedding(input_size, hidden_size)  # location embedding
        self.user_encoder = nn.Embedding(user_count, hidden_size)  # user embedding
        self.pref_encoder = nn.Embedding(1, hidden_size * 2)
        self.rnn = rnn_factory.create(hidden_size)
        self.project_matrix = nn.Linear(hidden_size, hidden_size * 2)  # 将user和location投影到同一片子空间
        self.fc = nn.Linear(2 * hidden_size, input_size)  # create outputs in length of locations

    def forward(self, x, t, s, y_t, y_s, h, active_user):
        seq_len, user_len = x.size()
        x_emb = self.encoder(x)  # (seq_len, user_len, hidden_size)
        out, h = self.rnn(x_emb, h)  # (seq_len, user_len, hidden_size)

        out_w = torch.zeros(seq_len, user_len, self.hidden_size, device=x.device)
        p_u = self.user_encoder(active_user)  # (1, user_len, hidden_size)
        p_u = p_u.view(user_len, self.hidden_size)  # (user_len, hidden_size)
        p_proj_u = torch.tanh(self.project_matrix(p_u))  # (user_len, hidden_size * 2)
        x_proj_emb = torch.tanh(self.project_matrix(x_emb).permute(1, 0, 2))  # (user_len, seq_len, hidden_size * 2)

        preference_emb = self.pref_encoder(torch.LongTensor([0]).to(x.device))
        user_loc_similarity = calculate_similarity(p_proj_u, x_proj_emb, preference_emb,
                                                   device=x.device)  # (user_len, seq_len)

        # compute weights per user
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
                w_j = w_j * user_loc_similarity[:, i].unsqueeze(1)  # (user_len, 1)
                sum_w += w_j
                out_w[i] = w_j * out[j]  # (user_len, hidden_size)
            out_w[i] /= sum_w

        # add user embedding:
        p_u = self.user_encoder(active_user)  # (1, user_len, hidden_size)
        p_u = p_u.view(user_len, self.hidden_size)  # (user_len, hidden_size)
        out_pu = torch.zeros(seq_len, user_len, 2 * self.hidden_size, device=x.device)
        for i in range(seq_len):
            out_pu[i] = torch.cat([out_w[i], p_u], dim=1)  # (user_len, hidden_size * 2)
        y_linear = self.fc(out_pu)  # (seq_len, user_len, input_size)

        # consider other users' embedding by user similarity
        out = out.permute(1, 0, 2)  # (user_len, seq_len, hidden_size) RNN的输出
        # others_out_w = torch.zeros(user_len, seq_len, self.user_count - 1, self.hidden_size, device=x.device)
        active_user = active_user.squeeze()  # (user_len, )
        user_similarity_matrix = torch.tensor(load_graph_data(pkl_filename='data/user_similarity_graph.pkl'))  # (user_count, user_count)
        final_out = []
        for k in range(user_len):
            others_out_w = torch.zeros(seq_len, self.user_count - 1, self.hidden_size, device=x.device)
            other_users = torch.ones(self.user_count)  # 对于一个active_user来说，其他用户都未曾真实拥有这条checkin轨迹
            current_user = int(active_user[k].item())  # 当前active_user所对应的user_id
            other_users[current_user] = 0  # 用1代表其他用户  从(user_count-1, 1)->(user_count-1)->(1, user_count-1)
            other_users_ids = torch.nonzero(other_users).squeeze().unsqueeze(0).to(x.device)  # 返回索引顺序0-cur_user-1, cur_user +1-user_count
            del other_users

            # 获取other users embedding
            p_others_u = self.user_encoder(other_users_ids)  # (1, user_count - 1, hidden_size)
            p_others_u = p_others_u.view(-1, self.hidden_size)  # (user_count - 1, hidden_size)
            p_proj_others_u = torch.tanh(self.project_matrix(p_others_u))  # (user_count - 1, hidden_size * 2)
            # (user_count - 1, seq_len, hidden_size * 2)
            x_proj_others_emb = x_proj_emb[k].unsqueeze(0).expand(self.user_count - 1, seq_len, self.hidden_size * 2)
            others_user_loc_similarity = calculate_similarity(p_proj_others_u, x_proj_others_emb, preference_emb,
                                                              device=x.device)  # (user_count - 1, seq_len)
            del other_users_ids, p_proj_others_u, x_proj_others_emb

            for i in range(seq_len):  # 对一个active_user的一条checkin 轨迹
                sum_w = torch.zeros(self.user_count - 1, 1, device=x.device)  # (user_count - 1, 1)
                for j in range(i + 1):
                    dist_t = (t[i][k] - t[j][k]).expand(1)  # 这里不需要.expand(self.user_count - 1),因为对其他users来说是一样的
                    dist_s = torch.norm(s[i][k] - s[j][k], dim=-1).expand(1)  # .expand(self.user_count - 1)
                    a_j = self.f_t(dist_t, 1)  # (1, 1)
                    b_j = self.f_s(dist_s, 1)
                    a_j = a_j.unsqueeze(1)  # (1, 1)
                    b_j = b_j.unsqueeze(1)
                    w_j = a_j * b_j + 1e-10  # small epsilon to avoid 0 division
                    w_j = w_j * others_user_loc_similarity[:, i].unsqueeze(1)  # (user_count - 1, 1)
                    sum_w += w_j
                    # (hidden_size, ) -> (user_count - 1, hidden_size) 指hidden state
                    current_u_out = out[k][j].unsqueeze(0).expand(self.user_count - 1, self.hidden_size)
                    others_out_w[i] = w_j * current_u_out  # (user_count - 1, hidden_size)
                others_out_w[i] /= sum_w
            # 已经获得了其他用户的输出和对应的embedding
            out_p_others_u = torch.zeros(seq_len, self.user_count - 1, 2 * self.hidden_size, device=x.device)
            for m in range(seq_len):  # (user_count - 1, hidden_size * 2)
                out_p_others_u[m] = torch.cat([others_out_w[m], p_others_u], dim=1)

            y_current_u = y_linear[:, k, :].unsqueeze(1)  # (seq_len, input_size) -> (seq_len, 1, input_size)
            similarity_current_u = user_similarity_matrix[current_user]  # 当前user与其他user的相似度 (user_count, )
            y_final_u = similarity_current_u[current_user] * y_current_u  # 当前user的输出 * 相似度
            for u in range(self.user_count - 1):
                # 计算某个inactive_user的输出
                y_other_u = self.fc(out_p_others_u[:, u, :]).unsqueeze(1)  # (seq_len, 1, input_size)
                if current_user == 0:  # 如果当前user是第一个(即索引为0),则u实际的索引应该是1, 2, 3, ..., user_count-1,即u+1!
                    y_final_u += similarity_current_u[u+1] * y_other_u
                elif current_user == self.user_count - 1:  # 如果当前user是最后一个(即索引为-1),则u实际的索引应该是u
                    y_final_u += similarity_current_u[u] * y_other_u
                else:  # 如果当前user位于中间,设为K,则u实际的索引应该是0,1,...K-1, K+1, K+2, ..., user_count-1
                    if u < current_user:
                        y_final_u += similarity_current_u[u] * y_other_u
                    else:
                        y_final_u += similarity_current_u[u+1] * y_other_u
            final_out.append(y_final_u)

        final_out = torch.cat(final_out, dim=1)  # (seq_len, user_len, input_size)
        return final_out, h

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
