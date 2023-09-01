import pickle
import numpy as np
import torch
import random
import torch.nn.functional as F
from math import radians, cos, sin, asin, sqrt
from scipy.sparse import csr_matrix, coo_matrix, identity, dia_matrix
import scipy.sparse as sp

seed = 0
global_seed = 0
torch.manual_seed(seed)


def load_graph_data(pkl_filename):
    graph = load_pickle(pkl_filename)  # list
    # graph = np.array(graph[0])
    return graph


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ': ', e)
        raise
    return pickle_data


def calculate_preference_similarity(m1, m2, pref):
    """
        m1: (user_len, hidden_size)
        m2：(user_len, seq_len, hidden_size)
        return: calculate the similarity between user and location, which means user's preference about location
    """
    user_len = m1.shape[0]
    seq_len = m2.shape[1]
    pref = pref.squeeze()  # (1, hidden_size) -> (hidden_size, )
    similarity = torch.zeros(user_len, seq_len, dtype=torch.float32)
    for i in range(user_len):
        v1 = m1[i]
        for j in range(seq_len):
            v2 = m2[i][j]
            similarity[i][j] = (1 + torch.cosine_similarity(v1 + pref, v2, dim=0).item()) / 2  # 归一化到[0, 1]

    return similarity


def compute_preference(m1, m2, pref):
    m1 = (m1 + pref).unsqueeze(1)
    s = m1 - m2
    sim = torch.exp(-(torch.norm(s, p=2, dim=-1)))
    return sim


# def calculate_friendship_similarity(u1, m2, pref, device):
#     """
#         u1: (1, hidden_size)
#         m2：(user_count - 1, hidden_size)
#         cur_u: 代表当前用户u1的实际索引id
#         return: calculate the similarity between users, which means user's similarity
#     """
#     user_len = m2.shape[0]
#     pref = pref.squeeze()
#     u1 = u1.squeeze()
#     similarity = torch.zeros(user_len, dtype=torch.float32).to(device)  # (user_count - 1, )
#     for u in range(user_len):
#         u2 = m2[u]
#         similarity[u] = (1 + torch.cosine_similarity(u1 + friend, u2, dim=0).item()) / 2  # 归一化到[0, 1]
#
#     return similarity


def get_user_static_preference(pref, locs):
    """
        pref: (user_len, seq_len)
        locs: (user_len, seq_len, hidden_size)
        return: 返回用户对于所访问POI的全局偏好
    """
    # pref = torch.softmax(pref, dim=1)  # (user_len, seq_len)
    # pref = pref.unsqueeze(2)  # (user_len, seq_len, 1)
    # user_preference = pref * locs  # (user_len, seq_len, hidden_size)
    # user_preference = user_preference.permute(1, 0, 2)  # (seq_len, user_len, hidden_size)
    user_len, seq_len = pref.shape[0], pref.shape[1]
    hidden_size = locs.shape[2]
    user_preference = torch.zeros(user_len, seq_len, hidden_size)
    for i in range(user_len):
        for j in range(seq_len):  # (hidden_size, )
            user_preference[i][j] = torch.sum(torch.softmax(pref[i, :j + 1], dim=0).unsqueeze(1) * locs[i, :j + 1],
                                              dim=0)
    user_preference = user_preference.permute(1, 0, 2)  # (seq_len, user_len, hidden_size)

    return user_preference


def sampling_prob(prob, label, num_neg):
    num_label, l_m = prob.shape[0], prob.shape[1]  # prob (batch_size, loc_count)
    init_label = torch.zeros(num_label, dtype=torch.int64)  # (batch_size, )
    init_prob = torch.zeros(size=(num_label, num_neg + 1))  # (batch_size, num_neg + 1)

    for batch in range(num_label):
        random_ig = random.sample(range(l_m), num_neg)  # (num_neg) from (0 -- l_max - 1)
        while label[batch].item() in random_ig:  # no intersection
            # print('循环查找')
            random_ig = random.sample(range(l_m), num_neg)

        # place the pos labels ahead and neg samples in the end
        for i in range(num_neg + 1):
            if i < 1:
                init_prob[batch, i] = prob[batch, label[batch]]
            else:
                init_prob[batch, i] = prob[batch, random_ig[i - 1]]

    global global_seed
    random.seed(global_seed)
    global_seed += 1

    return torch.FloatTensor(init_prob), torch.LongTensor(init_label)  # (batch_size, num_neg+1), (batch_size)


def bprLoss(pos, neg, target=1.0):
    loss = - F.logsigmoid(target * (pos - neg))
    return loss.mean()


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def top_transition_graph(transition_graph):
    graph = coo_matrix(transition_graph)
    data = graph.data
    row = graph.row
    threshold = 20

    for i in range(0, row.size, threshold):
        row_data = data[i: i + threshold]
        norm = row_data.max()
        row_data = row_data / norm
        data[i: i + threshold] = row_data

    return graph


def sparse_matrix_to_tensor(graph):
    graph = coo_matrix(graph)
    vaules = graph.data
    indices = np.vstack((graph.row, graph.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(vaules)
    shape = graph.shape
    graph = torch.sparse_coo_tensor(i, v, torch.Size(shape))

    return graph


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()

    return random_walk_mx  # D^-1 W


def calculate_reverse_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def log_string(log, string):
    """打印log"""
    log.write(string + '\n')
    log.flush()
    print(string)

if __name__ == '__main__':
    graph_path = 'data/user_similarity_graph.pkl'
    user_similarity_matrix = torch.tensor(load_graph_data(pkl_filename=graph_path))
    print(user_similarity_matrix[1])
    print('................')
    print(user_similarity_matrix[1][:10])
    count = 0
    # for i in range(user_similarity_matrix.shape[0]):
    #     for j in range(user_similarity_matrix.shape[1]):
    #         if user_similarity_matrix[i][j] > 0.01:  # 5747013, 即9.5%
    #             count += 1

    print('count: ', count)
