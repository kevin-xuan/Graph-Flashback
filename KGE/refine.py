# 因为final_train_triplets.txt包含了所有friend以及spatial关系,造成final_test_triplets.txt里没有这两种关系.
# 因此需要将训练集中的两种关系以8:2划分,并保留原先的数据文件
from constant import DATA_NAME, SCHEME

if __name__ == '__main__':
    new_train_triplets = './dataset/{}/{}_scheme{}/new_final_train_triplets.txt'.format(DATA_NAME, DATA_NAME, SCHEME)
    new_test_triplets = './dataset/{}/{}_scheme{}/new_final_test_triplets.txt'.format(DATA_NAME, DATA_NAME, SCHEME)
    f_new_train = open(new_train_triplets, 'w+')
    f_new_test = open(new_test_triplets, 'w+')

    train_friend_triplets = set()
    train_spatial_triplets = set()
    ratio = 0.8

    # 原封不动地把interact和temporal三元组写入新文件中
    with open('./dataset/{}/{}_scheme{}/final_test_triplets.txt'.format(DATA_NAME, DATA_NAME, SCHEME), 'r') as f_test:
        for line in f_test.readlines():
            tokens = tuple(line.strip('\n').split('\t'))
            h, t, r = tokens  # str
            f_new_test.write(h + '\t')
            f_new_test.write(t + '\t')
            f_new_test.write(r + '\n')

    with open('./dataset/{}/scheme_{}/final_train_triplets.txt', 'r').format(DATA_NAME, SCHEME) as f_train:
        for line in f_train.readlines():
            tokens = tuple(line.strip('\n').split('\t'))
            h, t, r = tokens  # str
            relation = int(tokens[2])
            if relation == 0:  # 原封不动的写入新文件中
                f_new_train.write(h + '\t')
                f_new_train.write(t + '\t')
                f_new_train.write(r + '\n')
            elif relation == 1:
                f_new_train.write(h + '\t')
                f_new_train.write(t + '\t')
                f_new_train.write(r + '\n')
            elif relation == 2:  # spatial
                if (t, h, r) not in train_spatial_triplets:  # 因为spatial和friend是对称的
                    train_spatial_triplets.add((h, t, r))
            else:  # friend
                if (t, h, r) not in train_friend_triplets:  # 因为spatial和friend是对称的
                    train_friend_triplets.add((h, t, r))
        print(len(train_spatial_triplets))  # 2258593
        print(len(train_friend_triplets))  # 27616

    train_spatial_triplets_len = int(len(train_spatial_triplets) * ratio)
    train_friend_triplets_len = int(len(train_friend_triplets) * ratio)

    count_spatial = 0
    count_friend = 0

    for elem_spatial in train_spatial_triplets:
        h, t, r = elem_spatial
        if count_spatial < train_spatial_triplets_len:
            f_new_train.write(h + '\t')
            f_new_train.write(t + '\t')
            f_new_train.write(r + '\n')

            f_new_train.write(t + '\t')
            f_new_train.write(h + '\t')
            f_new_train.write(r + '\n')
            count_spatial += 1
        else:
            f_new_test.write(h + '\t')
            f_new_test.write(t + '\t')
            f_new_test.write(r + '\n')

            f_new_test.write(t + '\t')
            f_new_test.write(h + '\t')
            f_new_test.write(r + '\n')
    print(count_spatial)
    for elem_friend in train_friend_triplets:
        h, t, r = elem_friend
        if count_friend < train_friend_triplets_len:
            f_new_train.write(h + '\t')
            f_new_train.write(t + '\t')
            f_new_train.write(r + '\n')

            f_new_train.write(t + '\t')
            f_new_train.write(h + '\t')
            f_new_train.write(r + '\n')
            count_friend += 1
        else:
            f_new_test.write(h + '\t')
            f_new_test.write(t + '\t')
            f_new_test.write(r + '\n')

            f_new_test.write(t + '\t')
            f_new_test.write(h + '\t')
            f_new_test.write(r + '\n')
    print(count_friend)
