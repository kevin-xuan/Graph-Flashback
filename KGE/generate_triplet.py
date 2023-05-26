#  先构造user-item，划分train/test集
#  定义relation2id.txt
#  根据user/item构造entity2id.txt
#  然后构造train/test 三元组
#  去掉重复三元组

import os
from setting import Setting
from dataloader import PoiDataloader
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm
from collections import defaultdict
from constant import DATA_NAME, SCHEME


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


def generate_train_test_checkin(train_file, test_file):
    if os.path.exists(train_file):
        print('train.txt and test.txt has existed!!!')
        return

    with open(train_file, 'w+') as f_train, open(test_file, 'w+') as f_test:
        for i, (user, loc) in enumerate(zip(users, pois)):
            train_thr = int(len(loc) * 0.8)
            train_locs = loc[: train_thr]
            test_locs = loc[train_thr:]
            train_locs.insert(0, user)
            test_locs.insert(0, user)

            for train_elem in train_locs:
                f_train.write(str(train_elem) + ' ')
            f_train.write('\n')
            for test_elem in test_locs:
                f_test.write(str(test_elem) + ' ')
            f_test.write('\n')
    print('Successfully generate train/test checkins!')


def generate_entity_file(entity2id_file):  # 构造entity2id文件
    if os.path.exists(entity2id_file):
        print('entity2id.txt has existed!!!')
        return
    with open(entity2id_file, 'w+') as f:
        # users_count = len(users)
        for i in range(users_count):
            f.write(str(i) + ' ')
            f.write(str(i) + ' ')
            f.write('\n')
        for value in poi2id.values():
            poi_id = value + users_count
            f.write(str(poi_id) + ' ')
            f.write(str(poi_id) + ' ')
            f.write('\n')
    print('Successfully generate entity2id.txt!')


def generate_train_test_triplets(train_file, train_triplets_file, friendship_file):  # 构造train/test 三元组
    f_train_triplets = open(train_triplets_file, 'w+')
    print('Construct interact relation and temporal relation......')
    with tqdm(total=users_count) as bar:
        with open(train_file, 'r') as f:
            for line in f.readlines():
                line = line.strip().split(' ')  # 以空格形式分隔
                user_id = line[0]  # str
                poi_ids = line[1:]  # poi2id字典中的org_id，在entity2id中对应id是org_id + users_count

                # 构建interact关系

                for poi_id in poi_ids:
                    poi_id = str(int(poi_id) + users_count)
                    f_train_triplets.write(user_id + '\t')
                    f_train_triplets.write(poi_id + '\t')
                    f_train_triplets.write('0' + '\n')  # 0代表interact relation

                # 构建temporal关系  相邻poi相连
                # print('Construct temporal relation......')
                for i in range(len(poi_ids) - 1):
                    poi_prev = str(int(poi_ids[i]) + users_count)
                    poi_next = str(int(poi_ids[i + 1]) + users_count)
                    if poi_prev != poi_next:
                        f_train_triplets.write(poi_prev + '\t')
                        f_train_triplets.write(poi_next + '\t')
                        f_train_triplets.write('1' + '\n')  # 1代表temporal relation
                bar.update(1)

    # 构建spatial关系  两个poi的距离小于距离阈值lambda_d，就相连
    print('Construct spatial relation......')
    pois_list = []
    for poi, coord in poi2gps.items():  # 生成元组列表, 即[(poi_1, coord_1), ...]
        pois_list.append((poi, coord))

    # 方案1
    if SCHEME == 1:
        lambda_d = 0.2  # 距离阈值为0.2千米
        with tqdm(total=len(pois_list)) as bar:
            for i in range(len(pois_list)):
                for j in range(i+1, len(pois_list)):
                    poi_prev, coord_prev = pois_list[i]
                    poi_next, coord_next = pois_list[j]
        
                    poi_prev = poi_prev + users_count  # poi在entity中所对应的实体集
                    poi_next = poi_next + users_count
                    lat_prev, lon_prev = coord_prev
                    lat_next, lon_next = coord_next
        
                    dist = haversine(lat_prev, lon_prev, lat_next, lon_next)
                    if dist <= lambda_d:
                        f_train_triplets.write(str(poi_prev) + '\t')
                        f_train_triplets.write(str(poi_next) + '\t')
                        f_train_triplets.write('2' + '\n')  # 2代表spatial relation
                        # spatial relation是对称的
                        f_train_triplets.write(str(poi_next) + '\t')
                        f_train_triplets.write(str(poi_prev) + '\t')
                        f_train_triplets.write('2' + '\n')
        
                bar.update(1)

    # 方案2
    else:
        lambda_d = 3  # 距离阈值为3千米, 再取top k, 即双重限制
        with tqdm(total=len(pois_list)) as bar:
            for i in range(len(pois_list)):
                loci_list = []
                for j in range(len(pois_list)):
                    poi_prev, coord_prev = pois_list[i]
                    poi_next, coord_next = pois_list[j]

                    poi_prev = poi_prev + users_count  # poi在entity中所对应的实体集
                    poi_next = poi_next + users_count
                    lat_prev, lon_prev = coord_prev
                    lat_next, lon_next = coord_next

                    dist = haversine(lat_prev, lon_prev, lat_next, lon_next)
                    if dist <= lambda_d and poi_prev != poi_next:
                        loci_list.append((poi_next, dist))  # 先是第一重限制, 这样可能会造成很多重复计算

                sort_list = sorted(loci_list, key=lambda x: x[1])  # 从小到大排序,距离越小,排名越靠前
                length = min(len(sort_list), 50)
                select_pois = sort_list[:length]  # 一般情况下, sort_list的长度肯定不止50, 取top 50  这是第二重限制
                for poi_entity, _ in select_pois:
                    f_train_triplets.write(str(poi_prev) + '\t')
                    f_train_triplets.write(str(poi_entity) + '\t')
                    f_train_triplets.write('2' + '\n')  # 2代表spatial relation
                    # spatial relation是对称的
                    f_train_triplets.write(str(poi_entity) + '\t')
                    f_train_triplets.write(str(poi_prev) + '\t')
                    f_train_triplets.write('2' + '\n')

                bar.update(1)

    # 构建friend关系  互为朋友的user相连  这个train/test会重复构造一次,可以选择生成一个friend_triplet文件,然后再将其内容放入train/test
    # 但因为数量很少,构造很快,所以放在一起
    print('Construct friend relation......')
    with open(friendship_file, 'r') as f_friend:
        for friend_line in f_friend.readlines():
            tokens = friend_line.strip('\n').split('\t')
            if user2id.get(int(tokens[0])) and user2id.get(int(tokens[1])):  # only focus on active users
                user_id1 = str(user2id.get(int(tokens[0])))
                user_id2 = str(user2id.get(int(tokens[1])))
                f_train_triplets.write(user_id1 + '\t')
                f_train_triplets.write(user_id2 + '\t')
                f_train_triplets.write('3' + '\n')  # 2代表friend relation
                # friend relation是对称的
                f_train_triplets.write(user_id2 + '\t')
                f_train_triplets.write(user_id1 + '\t')
                f_train_triplets.write('3' + '\n')
    f_train_triplets.close()


# 可能会重复添加triplet，所以要进行去重操作，得到最终train triplets
def filter_train_triplet(read_file, write_file):
    filter_set = set()
    print('Filter repeated triplets......')
    count = 0
    with open(read_file, 'r') as f_read, open(write_file, 'w+') as f_write:
        for f_read_line in f_read.readlines():
            count += 1
            f_read_line = f_read_line.strip('\n')
            if f_read_line not in filter_set:
                filter_set.add(f_read_line)
        for triplet in filter_set:
            f_write.write(triplet + '\n')
    print('Original triplets: ', count)
    print('Final triplets: ', len(filter_set))
    return filter_set


# 去重且保证test triplets与train triplets不同
def filter_test_triplet(read_file, write_file, train_filter_set):
    filter_set = set()
    print('Filter repeated triplets......')
    count = 0
    with open(read_file, 'r') as f_read, open(write_file, 'w+') as f_write:
        for f_read_line in f_read.readlines():
            count += 1
            f_read_line = f_read_line.strip('\n')
            if f_read_line not in filter_set and f_read_line not in train_filter_set:
                filter_set.add(f_read_line)
        for triplet in filter_set:
            f_write.write(triplet + '\n')
    print('Original triplets: ', count)
    print('Final triplets: ', len(filter_set))


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
    data_path = './dataset/{}/{}_scheme{}'.format(DATA_NAME, DATA_NAME, SCHEME)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    users_count = len(users)
    # generate train & test file

    train_file = os.path.join(data_path, 'train.txt')
    test_file = os.path.join(data_path, 'test.txt')
    entity2id_file = os.path.join(data_path, 'entity2id.txt')

    train_triplets = os.path.join(data_path, 'train_triplets.txt')
    test_triplets = os.path.join(data_path, 'test_triplets.txt')
    friendship_file = setting.friend_file

    final_train_triplets = os.path.join(data_path, 'final_train_triplets.txt')
    final_test_triplets = os.path.join(data_path, 'final_test_triplets.txt')

    print('Generate train/test checkins......')
    generate_train_test_checkin(train_file, test_file)  # 划分train/test check-ins
    print('Generate entity2id......')
    generate_entity_file(entity2id_file)
    print('Construct train triplets......')
    generate_train_test_triplets(train_file, train_triplets, friendship_file)  # 生成train/test三元组
    print('Construct test triplets......')
    generate_train_test_triplets(test_file, test_triplets, friendship_file)

    train_filter_triplets = filter_train_triplet(train_triplets, final_train_triplets)  # train三元组去重
    filter_test_triplet(test_triplets, final_test_triplets, train_filter_triplets)  # test三元组去重
