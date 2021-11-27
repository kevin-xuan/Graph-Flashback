from setting import Setting
from dataloader import PoiDataloader
import numpy as np
import os
from collections import defaultdict
import pickle


def construct_user_similarity(user_similarity_file):
    if os.path.exists(user_similarity_file):
        print('user_similarity_graph has been constructed!!!')
        return

    user_similarity_graph = np.ones((users_count, users_count), dtype=np.float32)
    user_loc = defaultdict(set)
    for i, (user, loc) in enumerate(zip(users, pois)):
        user_loc[user] = set(loc)  # user是raw data中org_id映射的map_id, 从0开始

    for j in range(users_count):
        for k in range(j + 1, users_count):
            user_similarity_graph[j][k] = len(user_loc[j].intersection(user_loc[k])) / len(user_loc[j])
            user_similarity_graph[k][j] = len(user_loc[k].intersection(user_loc[j])) / len(user_loc[k])

    # Normalized to [0, 1]
    diag_user = np.diag(np.sum(user_similarity_graph, axis=1))
    diag_user = np.power(diag_user, -1)
    diag_user[np.isinf(diag_user)] = 0.
    user_similarity_graph = np.matmul(diag_user, user_similarity_graph)

    with open(user_similarity_file, 'wb') as f:
        pickle.dump([user_similarity_graph], f, protocol=2)

    print('Successfully construct user similarity graph!')


def construct_loc_similarity(loc_similarity_file):
    if os.path.exists(loc_similarity_file):
        print('user_similarity_graph has been constructed!!!')
        return
    pass


if __name__ == '__main__':
    # parse settings
    setting = Setting()
    setting.parse()
    print(setting)

    # load dataset
    poi_loader = PoiDataloader(setting.max_users, setting.min_checkins)  # 0， 5*20+1
    poi_loader.read(setting.dataset_file)
    print('Active POI number: ', poi_loader.locations())  # 18737
    print('Active User number: ', poi_loader.user_count())  # 32510
    print('Total Checkins number: ', poi_loader.checkins_count())  # 1278274

    users = poi_loader.users
    pois = poi_loader.locs  # 二重列表，每个元素是user对应的checkin数据
    user2id = poi_loader.user2id
    poi2id = poi_loader.poi2id
    poi2gps = poi_loader.poi2gps
    users_count = len(users)

    user_file = './data/user_similarity_graph.pkl'
    loc_file = './data/loc_similarity_graph.pkl'

    construct_user_similarity(user_file)
    # construct_loc_similarity(loc_file)