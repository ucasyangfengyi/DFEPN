import collections

import numpy as np
import random
from typing import List
from utils.data import get_tsv_data
import torch


def random_sample_cls(sentences: List[str], labels: List[str], n_support: int, n_query: int, label: str):
    """
    Randomly samples Ns examples as support set and Nq as Query set
    """
    data = [sentences[i] for i, lab in enumerate(labels) if lab == label]
    perm = torch.randperm(len(data))
    idx = perm[:n_support]
    support = [data[i] for i in idx]
    idx = perm[n_support: n_support + n_query]
    query = [data[i] for i in idx]

    return support, query


def create_episode(data_dict,sup_data_dict, n_support, n_classes, n_query, n_gen_support,n_unlabeled=0):
    n_classes = min(n_classes, len(data_dict.keys()))
    rand_keys = np.random.choice(list(data_dict.keys()), n_classes, replace=False)
    assert min([len(val) for val in data_dict.values()]) >= n_support + n_query + n_unlabeled
    for key, val in data_dict.items():
        random.shuffle(val)
    xs = [[data_dict[k][i] for i in range(n_support)] for k in rand_keys]
    xq = [[data_dict[k][n_support + i] for i in range(n_query)] for k in rand_keys]

    xg = []
    if sup_data_dict != None:
        rand_keys = np.random.choice(list(sup_data_dict.keys()), n_gen_support, replace=False)
        for key, val in sup_data_dict.items():
            random.shuffle(val)
        xg = [[sup_data_dict[k][i] for i in range(2)] for k in rand_keys]


    episode = {'xs':xs,'xq':xq,'xg':xg}

    # episode = {
    #     "xs": [
    #         [data_dict[k][i] for i in range(n_support)] for k in rand_keys
    #     ],
    #     "xq": [
    #         [data_dict[k][n_support + i] for i in range(n_query)] for k in rand_keys
    #     ]
    # }

    if n_unlabeled:
        episode['xu'] = [
            item for k in rand_keys for item in data_dict[k][n_support + n_query:n_support + n_query + 10]
        ]
    return episode

def create_snips_episode(target_support_dict,target_query_dict,sup_data_dict, n_support, n_classes, n_query, n_gen_support,n_unlabeled=0):
    n_classes = min(n_classes, len(target_support_dict.keys()))
    # 生成support set
    rand_keys = np.random.choice(list(target_support_dict.keys()), n_classes, replace=False)
    for key, val in target_support_dict.items():
        random.shuffle(val)
    xs = [[target_support_dict[k][i] for i in range(n_support)] for k in rand_keys]
    # 生成query set
    for key, val in target_query_dict.items():
        random.shuffle(val)
    xq = [[target_query_dict[k][i] for i in range(n_query)] for k in rand_keys]

    # 生成辅助数据
    xg = []
    rand_keys = np.random.choice(list(sup_data_dict.keys()), n_gen_support, replace=True)
    for k in rand_keys:
        for key, val in sup_data_dict.items():
            random.shuffle(val)
        # xg = [[sup_data_dict[k][i] for i in range(2)] for k in rand_keys]
        xg.append([sup_data_dict[k][i] for i in range(2)])
    episode = {'xs':xs,'xq':xq,'xg':xg}


    # episode = {
    #     "xs": [
    #         [data_dict[k][i] for i in range(n_support)] for k in rand_keys
    #     ],
    #     "xq": [
    #         [data_dict[k][n_support + i] for i in range(n_query)] for k in rand_keys
    #     ]
    # }


    return episode

def create_snips_noneps(target_support_dict,target_query_dict,sup_data_dict, n_support, n_classes, n_query, n_gen_support,n_unlabeled=0):
    n_classes = min(n_classes, len(target_support_dict.keys()))
    # 生成support set
    rand_keys = np.random.choice(list(target_support_dict.keys()), n_classes, replace=False)
    for key, val in target_support_dict.items():
        random.shuffle(val)
    xs = [[target_support_dict[k][i] for i in range(n_support)] for k in rand_keys]
    # 生成query set
    for key, val in target_query_dict.items():
        random.shuffle(val)
    xq = [[target_query_dict[k][i] for i in range(len(target_query_dict[k]))] for k in rand_keys]

    # 生成辅助数据
    xg = []
    rand_keys = np.random.choice(list(sup_data_dict.keys()), n_gen_support, replace=True)
    for k in rand_keys:
        for key, val in sup_data_dict.items():
            random.shuffle(val)
        # xg = [[sup_data_dict[k][i] for i in range(2)] for k in rand_keys]
        xg.append([sup_data_dict[k][i] for i in range(2)])
    # rand_keys = np.random.choice(list(sup_data_dict.keys()), n_gen_support, replace=False)
    # for key, val in sup_data_dict.items():
    #     random.shuffle(val)
    # xg = [[sup_data_dict[k][i] for i in range(2)] for k in rand_keys]
    episode = {'xs':xs,'xq':xq,'xg':xg}
    return episode



def create_clinc_episode(target_query_dict,target_support_dict,sup_data_dict,
                        n_support, n_classes, n_query, n_gen_support,n_unlabeled=0):
    n_classes = min(n_classes, len(target_support_dict.keys()))
    # 生成support set
    rand_keys = np.random.choice(list(target_support_dict.keys()), n_classes, replace=False)
    for key, val in target_support_dict.items():
        random.shuffle(val)
    xs = [[target_support_dict[k][i] for i in range(n_support)] for k in rand_keys]
    # 生成query set
    # for key, val in target_query_dict.items():
    #     random.shuffle(val)

    xq_list = []
    a = len(target_query_dict[rand_keys[0]]) / n_query
    for temp in range(int(a)):
        xq = [[target_query_dict[k][i+temp*n_query] for i in range(n_query)] for k in rand_keys]
        xq_list.append(xq)

    # xq = [[target_query_dict[k][i] for i in range(n_query)] for k in rand_keys]

    # 生成辅助数据
    rand_keys = np.random.choice(list(sup_data_dict.keys()), n_gen_support, replace=False)
    for key, val in sup_data_dict.items():
        random.shuffle(val)
    xg = [[sup_data_dict[k][i] for i in range(2)] for k in rand_keys]

    episode_list = []
    for i in range(len(xq_list)):
        episode = {'xs':xs,'xq':xq_list[i],'xg':xg}
        episode_list.append(episode)
    return episode_list