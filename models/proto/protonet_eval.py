import sys
sys.path.append('/opt/yfy/PythonProjects/DFEPN')
from models.DTN.bert_classifier import BertClassifier
from models.DTN.nlue import GeneratorNet
from models.proto.protonet_snips import ProtoNet
import json
import argparse
# from models.encoders.bert_encoder import BERTEncoder
from utils.data import get_jsonl_data
from utils.python import now, set_seeds
import random
import collections
import os
from typing import List, Dict
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from torch.autograd import Variable
import warnings
import logging
from utils.few_shot import create_episode,create_snips_episode,create_snips_noneps
from utils.my_math import euclidean_dist, cosine_similarity
# from models.generator.generator import GeneratorNet
import tqdm
from copy import deepcopy

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.simplefilter('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
raw_gen_support = 25
# np.random.seed(raw_gen_support)

def raw_data_to_labels_dict(data, shuffle=True):
    labels_dict = collections.defaultdict(list)
    for item in data:
        labels_dict[item['label']].append(item["sentence"])
    labels_dict = dict(labels_dict)
    if shuffle:
        for key, val in labels_dict.items():
            random.shuffle(val)
    return labels_dict

def eps_test(target_query_dict, target_support_data_dict, sup_data_dict,
             tgt, n_episodes, n_classes, n_query, model):
    accuracies = list()
    losses = list()
    if tgt == 'novel':
        n_classes = n_classes
    elif tgt == 'joint':
        n_classes = 2 * n_classes
    for i in tqdm.tqdm(range(n_episodes)):
        episode = create_snips_episode(
            target_query_dict=target_query_dict,
            target_support_dict = target_support_data_dict,
            sup_data_dict=sup_data_dict,
            n_support=n_support,
            n_classes=n_classes,
            n_query=n_query,
            n_gen_support=raw_gen_support,
        )
        loss, loss_dict = model.loss(episode)
        accuracies.append(loss_dict["acc"])
        losses.append(loss_dict["loss"])
    return {
        "loss": np.mean(losses),
        "acc": np.mean(accuracies)
    }

def noneps_test(target_query_dict, target_support_data_dict, sup_data_dict,
             tgt, n_episodes, n_classes, n_query, model):
    correct_num = 0
    length = 0
    if tgt == 'novel':
        n_classes = 2
    elif tgt == 'joint':
        n_classes = 7

    all_episode = create_snips_noneps(
        target_query_dict=target_query_dict,
        target_support_dict = target_support_data_dict,
        sup_data_dict=sup_data_dict,
        n_support=n_support,
        n_classes=n_classes,
        n_query=n_query,
        n_gen_support=raw_gen_support,
    )
    batch_size = 40
    # 用于画图使用
    # xs = all_episode['xs']
    # xg = all_episode['xg']
    # xq = []
    # target_inds = []
    # for i in range(n_classes):
    #     xq.append(all_episode['xq'][i][:30])
    #     target_inds += [i] * 30
    # episode = {'xs': xs, 'xq': xq, 'xg': xg}
    # correct_num += model.loss(episode,target_inds)

    for i in range(n_classes):
        xs = all_episode['xs']
        xg = all_episode['xg']
        batch = int(len(all_episode['xq'][i]) / batch_size)
        for j in range(batch+1):
            if len(all_episode['xq'][i]) > (j+1)*batch_size:
                xq = [all_episode['xq'][i][j*batch_size : (j+1)*batch_size]]
            else:
                xq = [all_episode['xq'][i][j * batch_size : len(all_episode['xq'][i])]]
            target_inds = [torch.tensor(i)] * len(xq[0])
            if target_inds != []:
                target_inds = torch.stack(target_inds)
                target_inds = target_inds.to(device)
                episode = {'xs': xs, 'xq': xq, 'xg': xg}
                correct_num += model.loss(episode,target_inds)
        length += len(all_episode['xq'][i])
    return {
        "correct_num": correct_num,
        'length':length,
        "acc": correct_num/length
    }




if __name__ == '__main__':
    nlue_id = 3
    n_support = 1
    # tgt = 'joint'
    n_episodes = 1000
    n_classes = 2
    n_query = 5
    set_seeds(5)
    # trans_dataset = '/opt/yfy/PythonProjects/DFEPN/dataset/trans_dataset/NLUE/KFold_' + str(
    #     nlue_id) + '/'
    # model_name_or_path = '/opt/yfy/PythonProjects/DFEPN/language_modeling/transformer_models/nlue/kfold_1/fine-tuned/'

    trans_dataset = '/opt/yfy/PythonProjects/DFEPN/dataset/trans_dataset/SNIPS/'
    model_name_or_path = '/opt/yfy/PythonProjects/DFEPN/language_modeling/transformer_models/snips/fine-tuned/'
    metric = "cosine"  # cosine or euclidean
    bert_classifier = BertClassifier(model_name_or_path, 64).to(device)
    # bert_classifier.load_state_dict(torch.load('../DTN/save/mynet.nlue_1.1_shot.classifier.finetune.pkl'))
    generator = GeneratorNet().to(device)
    # generator.load_state_dict(torch.load('../DTN/save/mynet.nlue_1.1_shot.generator.finetune.pkl'))

    # bert = BertClassifier(model_name_or_path,64).to(device)
    # generator = GeneratorNet().to(device)
    protonet = ProtoNet(encoder=bert_classifier, generator=generator, metric=metric).to(device)
    # protonet.load_state_dict(torch.load('save/1_shot_snips_mynet.pkl.joint.'+str(50)+'sup'))
    protonet.load_state_dict(torch.load('save/1_shot_snips_.woquery.pkl.novel.25sup'))

    protonet.eval()

    tgt = 'joint'
    sup_data_path = trans_dataset + 'auxiliary_samples.jsonl'
    support_n_shots_data_path = trans_dataset + 'support_' + str(n_support) + '_shots_' + tgt + '_data.jsonl'
    test_data_path = trans_dataset + 'test_' + tgt + '_data.jsonl'
    target_query = get_jsonl_data(test_data_path)
    target_query_dict = raw_data_to_labels_dict(target_query, shuffle=True)
    target_support_data = get_jsonl_data(support_n_shots_data_path)
    target_support_data_dict = raw_data_to_labels_dict(target_support_data, shuffle=True)
    sup_data = get_jsonl_data(sup_data_path)
    sup_data_dict = raw_data_to_labels_dict(sup_data, shuffle=True)

    # for sup_key in sup_data_dict:
    #     del target_query_dict[sup_key]
    #     del target_support_data_dict[sup_key]
    #     print(sup_key)

    eps_sj = eps_test(target_query_dict, target_support_data_dict, sup_data_dict,
             tgt, n_episodes, n_classes, n_query, protonet)
    noneps_sj = noneps_test(target_query_dict, target_support_data_dict, sup_data_dict,
                    tgt, n_episodes, n_classes, n_query, protonet)


    tgt = 'novel'
    sup_data_path = trans_dataset + 'auxiliary_samples.jsonl'
    support_n_shots_data_path = trans_dataset + 'support_' + str(n_support) + '_shots_' + tgt + '_data.jsonl'
    test_data_path = trans_dataset + 'test_' + tgt + '_data.jsonl'
    target_query = get_jsonl_data(test_data_path)
    target_query_dict = raw_data_to_labels_dict(target_query, shuffle=True)
    target_support_data = get_jsonl_data(support_n_shots_data_path)
    target_support_data_dict = raw_data_to_labels_dict(target_support_data, shuffle=True)
    sup_data = get_jsonl_data(sup_data_path)
    sup_data_dict = raw_data_to_labels_dict(sup_data, shuffle=True)
    eps_sn = eps_test(target_query_dict, target_support_data_dict, sup_data_dict,
                    tgt, n_episodes, n_classes, n_query, protonet)
    noneps_sn = noneps_test(target_query_dict, target_support_data_dict, sup_data_dict,
                 tgt, n_episodes, n_classes, n_query, protonet)
    print(noneps_sj)
    print(noneps_sn)
    print(eps_sj)
    print(eps_sn)
