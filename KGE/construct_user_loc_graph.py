import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix
import argparse

device = torch.device('cuda', 0)

def projection_transH(original, norm):
    return original - torch.sum(original * norm, dim=len(original.size()) - 1, keepdim=True) * norm


def projection_transR(original, proj_matrix):  # (batch, k_dim)   (batch, k_dim * dim) 
    ent_embedding_size = original.shape[1]
    rel_embedding_size = proj_matrix.shape[1] // ent_embedding_size
    original = original.view(-1, ent_embedding_size, 1)  # (batch, k_dim, 1)
    proj_matrix = proj_matrix.view(-1, rel_embedding_size, ent_embedding_size)  # (batch, dim, k_dim)
    return torch.matmul(proj_matrix, original).view(-1, rel_embedding_size)  # (batch, dim, 1) -> (batch, dim)


def calculate_score(h_e, t_e, rel, model_type, L1_flag=True, norm=None, proj=None):
    if model_type == "transe":
        if L1_flag:
            score = torch.exp(-torch.sum(torch.abs(h_e + rel - t_e), 1))
        else:
            score = torch.exp(-torch.sum(torch.abs(h_e + rel - t_e) ** 2, 1))

    elif model_type == "transh":
        proj_h_e = projection_transH(h_e, norm)
        proj_t_e = projection_transH(t_e, norm)
        if L1_flag:
            score = torch.exp(-torch.sum(torch.abs(proj_h_e + rel - proj_t_e), 1))
        else:
            score = torch.exp(-torch.sum(torch.abs(proj_h_e + rel - proj_t_e) ** 2, 1))

    else:
        proj_h_e = projection_transR(h_e, proj)
        proj_t_e = projection_transR(t_e, proj)
        if L1_flag:
            score = torch.exp(-torch.sum(torch.abs(proj_h_e + rel - proj_t_e), 1))
        else:
            score = torch.exp(-torch.sum(torch.abs(proj_h_e + rel - proj_t_e) ** 2, 1))

    return score


def construct_transition_graph(args, filename, user_encoder, loc_encoder, temporal_preference, norm=None, proj=None):
    loc_count = args.loc_count
    user_count = args.user_count

    threshold = args.threshold
    L1_flag = args.L1_flag
    model_type = args.model_type

    bar = tqdm(total=user_count)
    bar.set_description('Construct User-POI Graph')

    transition_graph = lil_matrix((user_count, loc_count), dtype=np.float32)  # directed graph
    for i in range(user_count):
        h_e = user_encoder(torch.LongTensor([i]).to(device))
        t_list = list(range(loc_count))
        t_e = loc_encoder(torch.LongTensor(t_list).to(device))

        transition_vector = calculate_score(h_e, t_e, temporal_preference, model_type, L1_flag, norm, proj)
        indices = torch.argsort(transition_vector, descending=True)[:threshold]  # top_k
        norm = torch.max(transition_vector[indices])
        for index in indices:
            index = index.item()
            transition_graph[i, index] = (transition_vector[index] / norm).item()

        bar.update(1)
    bar.close()

    with open(filename, 'wb') as f:
        pickle.dump(transition_graph, f, protocol=2)


def construct_friend_graph(args, user_encoder, loc_encoder, friend_preference, norm=None, proj=None, friend_flag=True):
    loc_count = args.loc_count
    user_count = args.user_count

    threshold = args.threshold
    L1_flag = args.L1_flag
    model_type = args.model_type

    bar = tqdm(total=user_count)
    if friend_flag:
        bar.set_description('Construct User friend Graph')
    else:
        bar.set_description('Construct User interact Graph')

    friend_graph = lil_matrix((user_count, user_count), dtype=np.float32)  
    for i in range(user_count):
        if friend_flag:
            h_e = user_encoder(torch.LongTensor([i]).to(device))
            t_list = list(range(user_count))
            t_e = user_encoder(torch.LongTensor(t_list).to(device))
            indices = torch.LongTensor(t_list[:i] + t_list[i + 1:]).to(device)
            transition_vector = calculate_score(h_e, t_e, friend_preference, model_type, L1_flag, norm, proj)
            transition_vector_a = torch.index_select(transition_vector, 0, indices)  # [0, 1, ..., i-1, i+1, i+2, ...]
            indices = torch.argsort(transition_vector_a, descending=True)[:threshold]  # top_k
            norm = torch.max(transition_vector_a[indices])
            for index in indices:
                index = index.item()
                if index < i:
                    pass
                else:
                    index += 1
                friend_graph[i, index] = (transition_vector[index] / norm).item()
        else:
            h_e = user_encoder(torch.LongTensor([i]).to(device))
        bar.update(1)
    bar.close()

    return friend_graph


def merge_graph(filename, friend_graph, interact_graph):
    friend_graph = csr_matrix(friend_graph)
    interact_graph = csr_matrix(interact_graph)
    friend_graph += interact_graph

    with open(filename, 'wb') as f:
        pickle.dump(friend_graph, f, protocol=2)


def get_parser():
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument("--model_type", default="transe", type=str, help="which KGE method to use")
    parser.add_argument("--dataset", default="foursquare", type=str, help="which dataset to use")
    parser.add_argument("--pretrain_model", default="../data/foursquare_scheme2/foursquare-transe-1641035874.ckpt", type=str, help="pretrained model")
    parser.add_argument("--version", default="scheme2", type=str, help="which version of KG")
    parser.add_argument("--threshold", default=20, type=int, help="top_K")
    parser.add_argument("--user_count", default=45343, type=int, help="number of user")  # 45343  7768
    parser.add_argument("--loc_count", default=68879, type=int, help="number of POI")  # 68879  106994
    parser.add_argument("--L1_flag", default=True, type=bool, help="whether to use L1 or L2 norm")
    parser.add_argument("--loc_graph", default=False, type=bool, help="whether to construct POI or user graph")
    parser.add_argument("--loc_spatial", default=False, type=bool, help="whether to construct temporal or spatial POI graph")
    parser.add_argument("--friend_flag", default=False, type=bool, help="whether to construct friend graph")
    args = parser.parse_args()
    return args


def main():
    args = get_parser()
    pretrain_model = torch.load(args.pretrain_model, map_location=lambda storage, loc: storage)
    user_count = args.user_count
    graph_type = 'user-loc'
    # if args.loc_graph:
    #     graph_type = 'loc'
    #     if args.loc_spatial:
    #         graph_type = graph_type + '_spatial'
    #     else:
    #         graph_type = graph_type + '_temporal'
    # else:
    #     graph_type = 'user'
    prefix = 'POI_graph'
    if not os.path.exists('./' + prefix):
        os.mkdir('./' + prefix)
    graph_file = './' + prefix + '/' + args.dataset + '_' + args.version + '_' + args.model_type + '_' + graph_type + '_' + str(
        args.threshold) + '.pkl'

    print(graph_file)

    user_encoder = nn.Embedding.from_pretrained(
        pretrain_model['model_state_dict']['ent_embeddings.weight'][:user_count]).to(device)
    loc_encoder = nn.Embedding.from_pretrained(
        pretrain_model['model_state_dict']['ent_embeddings.weight'][user_count:]).to(device)
    rel_encoder = nn.Embedding.from_pretrained(pretrain_model['model_state_dict']['rel_embeddings.weight']).to(device)

    interact_preference = rel_encoder(torch.LongTensor([0]).to(device))
    # temporal_preference = rel_encoder(torch.LongTensor([1]).to(device))
    # spatial_preference = rel_encoder(torch.LongTensor([2]).to(device))
    # friend_preference = rel_encoder(torch.LongTensor([3]).to(device))

    if args.model_type == "transh":
        norm_encoder = nn.Embedding.from_pretrained(pretrain_model['model_state_dict']['norm_embeddings.weight']).to(
            device)

        norm_interact = norm_encoder(torch.LongTensor([0]).to(device))
        # norm_temporal = norm_encoder(torch.LongTensor([1]).to(device))
        # norm_spatial = norm_encoder(torch.LongTensor([2]).to(device))
        # norm_friend = norm_encoder(torch.LongTensor([3]).to(device))
        construct_transition_graph(args, graph_file, user_encoder, loc_encoder, interact_preference, norm=norm_interact)
        # if args.loc_graph:
        #     if args.loc_spatial:
        #         construct_transition_graph(args, graph_file, loc_encoder, spatial_preference, norm=norm_spatial)
        #     else:
        #         construct_transition_graph(args, graph_file, loc_encoder, temporal_preference, norm=norm_temporal)
        # else:
        #     friend_graph = construct_friend_graph(args, graph_file, user_encoder, friend_preference, norm=norm_friend)
        #     interact_graph = construct_friend_graph(args, graph_file, user_encoder, interact_preference,
        #                                             norm=norm_interact)

    elif args.model_type == "transr":
        proj_encoder = nn.Embedding.from_pretrained(pretrain_model['model_state_dict']['proj_embeddings.weight']).to(
            device)

        proj_interact = proj_encoder(torch.LongTensor([0]).to(device))
        # proj_temporal = proj_encoder(torch.LongTensor([1]).to(device))
        # proj_spatial = proj_encoder(torch.LongTensor([2]).to(device))
        # proj_friend = proj_encoder(torch.LongTensor([3]).to(device))
        construct_transition_graph(args, graph_file, user_encoder, loc_encoder, interact_preference, proj=proj_interact)
        # if args.loc_graph:
        #     if args.loc_spatial:
        #         construct_transition_graph(args, graph_file, loc_encoder, spatial_preference, proj=proj_spatial)
        #     else:
        #         construct_transition_graph(args, graph_file, loc_encoder, temporal_preference, proj=proj_temporal)
        # else:
        #     friend_graph = construct_friend_graph(args, user_encoder, loc_encoder, friend_preference, proj=proj_friend, friend_flag=True)
        #     interact_graph = construct_friend_graph(args, user_encoder, loc_encoder, interact_preference, proj=proj_interact, friend_flag=False)
    else:
        construct_transition_graph(args, graph_file, user_encoder, loc_encoder, interact_preference)
        # if args.loc_graph:
        #     if args.loc_spatial:
        #         construct_transition_graph(args, graph_file, loc_encoder, spatial_preference)
        #     else:
        #         construct_transition_graph(args, graph_file, loc_encoder, temporal_preference)
        # else:
        #     construct_friend_graph(args, graph_file, user_encoder, friend_preference, interact_preference)


if __name__ == '__main__':
    main()
