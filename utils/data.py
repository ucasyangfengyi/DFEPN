import sys
sys.path.append('/opt/yfy/PythonProjects/DFEPN')

import os
import json
from typing import List, Dict
import random
import numpy as np
import collections
import itertools
'''
    将原始的数据集转换为jsonl的形式，例子：
    {"global_ix": 8843, "sentence": "ai, routing number for my b of a checking account", "label": "routing"}
    get_jsonl_data 从jsonl文件中读取数据
'''

def get_jsonl_data(jsonl_path: str):
    out = list()
    with open(jsonl_path, 'r') as file:
        for line in file:
            j = json.loads(line.strip())
            out.append(j)
    return out


def write_jsonl_data(jsonl_data: List[Dict], jsonl_path: str, force=False):
    if os.path.exists(jsonl_path) and not force:
        raise FileExistsError
    with open(jsonl_path, 'w') as file:
        for line in jsonl_data:
            file.write(json.dumps(line, ensure_ascii=False) + '\n')


def get_tsv_data(tsv_path: str, label: str = None):
    out = list()
    with open(tsv_path, "r") as file:
        for line in file:
            line = line.strip().split('\t')
            out.append({
                "sentence": line[1],
                "label":  line[0]
            })
    return out

class ClincLoader():
    def __init__(self):
        super(ClincLoader,self).__init__()
        self.ClincAllPath = '../dataset/raw_dataset/clinc150/all/'
        self.ClincBankPath = '../dataset/raw_dataset/clinc150/oos/'
        self.banklabels,self.trainlabels = self.load_bank_label()
    def load_data(self,istrain=True) -> List[Dict]:
        train_data = list()
        test_data = list()
        dev_data = list()

        if istrain:
            filepath = self.ClincAllPath
            labels = self.trainlabels
        else:
            filepath = self.ClincBankPath
            labels = self.banklabels

        with open(filepath + 'train/label','r') as fr1:
            with open(filepath + 'train/seq.in','r') as fr2:
                for line1,line2 in zip(fr1,fr2):
                    label = line1.strip()
                    sentence = line2.strip()
                    if label in labels:
                        train_data.append(
                            {
                                "sentence": sentence,
                                "label": label
                            }
                        )
        with open(filepath + 'dev/label','r') as fr1:
            with open(filepath + 'dev/seq.in','r') as fr2:
                for line1,line2 in zip(fr1,fr2):
                    label = line1.strip()
                    sentence = line2.strip()
                    if label in labels:
                        dev_data.append(
                            {
                                "sentence": sentence,
                                "label": label
                            }
                        )
        with open(filepath + 'test/label','r') as fr1:
            with open(filepath + 'test/seq.in','r') as fr2:
                for line1,line2 in zip(fr1,fr2):
                    label = line1.strip()
                    sentence = line2.strip()
                    if label in labels:
                        test_data.append(
                            {
                                "sentence": sentence,
                                "label": label
                            }
                        )
        return train_data,test_data,dev_data
    def load_bank_label(self):
        banklabels = []
        with open(self.ClincBankPath + 'train/label','r') as fr:
            for line in fr:
                label = line.strip()
                banklabels.append(label)
        banklabels = list(set(banklabels))
        alllabels = []
        with open(self.ClincAllPath + 'train/label','r') as fr:
            for line in fr:
                label = line.strip()
                alllabels.append(label)
        alllabels = list(set(alllabels))
        trainlabels = list(set(alllabels) - set(banklabels))
        return banklabels,trainlabels



def create_clinc_episode(target_support_dict,n_support, n_classes):
    few_shot_train = list()
    n_classes = min(n_classes, len(target_support_dict.keys()))
    # 生成support set
    rand_keys = np.random.choice(list(target_support_dict.keys()), n_classes, replace=False)
    for key, val in target_support_dict.items():
        random.shuffle(val)
    xs = [[target_support_dict[k][i] for i in range(n_support)] for k in rand_keys]
    for k in rand_keys:
        for i in range(n_support):
            sentence = target_support_dict[k][i]
            label = k
            few_shot_train.append(
                {
                    "sentence": sentence,
                    "label": label
                }
            )
    return few_shot_train
def raw_data_to_labels_dict(data, shuffle=True):
    labels_dict = collections.defaultdict(list)
    for item in data:
        labels_dict[item['label']].append(item["sentence"])
    labels_dict = dict(labels_dict)
    if shuffle:
        for key, val in labels_dict.items():
            random.shuffle(val)
    return labels_dict

def split_train_support(train_jsonl):
    data = get_jsonl_data(train_jsonl)
    sup_data = data[:4000]
    meta_data = data[4000:]
    return sup_data,meta_data

def splitDict(d,n):
    i = iter(d.items())
    d1 = dict(itertools.islice(i, n))   # grab first n items
    d2 = dict(i)                        # grab the rest
    return d1, d2

def split_train_seen_data(train_jsonl):
    train_data = get_jsonl_data(train_jsonl)
    train_data_dict = raw_data_to_labels_dict(train_data)
    support_data_dict, train_data_dict = splitDict(train_data_dict, 1)
    support_data_dict_jsonl = list()
    train_data_dict_jsonl = list()
    for key in support_data_dict:
        label = key
        for j in support_data_dict[key]:
            sentence = j
            support_data_dict_jsonl.append({
                            "sentence": sentence,
                            "label": label
                        })
    for key in train_data_dict:
        label = key
        for j in train_data_dict[key]:
            sentence = j
            train_data_dict_jsonl.append({
                            "sentence": sentence,
                            "label": label
                        })

    return support_data_dict_jsonl, train_data_dict_jsonl

if __name__ == '__main__':
    # trans_dataset = '../dataset/trans_dataset/clinc150/banking/'
    # train_jsonl = trans_dataset + 'train.jsonl'
    # sup_data,meta_data = split_train_support(train_jsonl)
    # write_jsonl_data(sup_data,trans_dataset + 'sup_data.jsonl')
    # write_jsonl_data(meta_data,trans_dataset + 'meta_data.jsonl')

    # file_path = '../dataset/trans_dataset/clinc150/banking/target.train.jsonl'
    # train_data = get_jsonl_data(file_path)
    # train_data_dict = raw_data_to_labels_dict(train_data, shuffle=True)
    # few_shot_train = create_clinc_episode(train_data_dict, n_support=5, n_classes=15)
    # write_jsonl_data(few_shot_train, trans_dataset + '15way5shot.target.train.jsonl')
    # print()

    # 生成clinc150的训练数据集
    # clincloader = ClincLoader()
    # trans_dataset = '../dataset/trans_dataset/clinc150/oos/'
    # train_data, test_data, dev_data = clincloader.load_data(istrain=False)
    # write_jsonl_data(train_data,trans_dataset+'oos.train.jsonl')
    # write_jsonl_data(test_data, trans_dataset + 'oos.test.jsonl')
    # write_jsonl_data(dev_data, trans_dataset + 'oos.dev.jsonl')
    #
    #
    # finetune_train_file = trans_dataset + 'train.txt'
    # finetune_dev_file = trans_dataset + 'dev.txt'
    # with open(finetune_train_file,'w',encoding='utf-8') as fw:
    #     for line in train_data:
    #         fw.write(line['sentence'])
    #         fw.write('\n')
    # with open(finetune_dev_file,'w',encoding='utf-8') as fw:
    #     for line in dev_data:
    #         fw.write(line['sentence'])
    #         fw.write('\n')
    # print()
    # train_data, test_data, dev_data = clincloader.load_data(istrain=False)
    # write_jsonl_data(train_data, trans_dataset + 'target.train.jsonl')
    # write_jsonl_data(test_data, trans_dataset + 'target.test.jsonl')
    # write_jsonl_data(dev_data, trans_dataset + 'target.dev.jsonl')


    '''
    # 从sample原来的jsonl数据集转换成原始数据集，用于finetune
    # file = '../dataset/trans_dataset/sample/valid.jsonl'
    # target_file = '../dataset/raw_dataset/sample/valid.txt'
    # file_data = get_jsonl_data(file)
    # with open(target_file,'w',encoding='utf-8') as fw:
    #     for line in file_data:
    #         fw.write(line['sentence'])
    #         fw.write('\n')
    #
    # print()
    '''

    # SNIPS or NLUE
    # train_seen_dataset = '../dataset/raw_dataset/SNIPS/train_seen.tsv'
    # support_1_shots_joint_dataset = '../dataset/raw_dataset/SNIPS/support_5_shots_joint'
    # support_1_shots_novel_dataset = '../dataset/raw_dataset/SNIPS/support_5_shots_novel'
    # test_novel_dataset = '../dataset/raw_dataset/SNIPS/test_novel.tsv'
    # test_joint_dataset = '../dataset/raw_dataset/SNIPS/test_joint.tsv'
    #
    trans_dataset = '../dataset/trans_dataset/SNIPS/'
    train_seen_data = trans_dataset + 'train_seen_data.jsonl'
    support_data_dict_jsonl, train_data_dict_json = split_train_seen_data(train_seen_data)
    write_jsonl_data(support_data_dict_jsonl,trans_dataset + 'auxiliary_samples.jsonl')
    write_jsonl_data(train_data_dict_json,trans_dataset + 'base_samples.jsonl')
    #
    # train_seen_data = get_tsv_data(train_seen_dataset)
    # support_1_shots_novel_data = get_tsv_data(support_1_shots_novel_dataset)
    # support_1_shots_joint_data = get_tsv_data(support_1_shots_joint_dataset)
    # test_novel_data = get_tsv_data(test_novel_dataset)
    # test_joint_data = get_tsv_data(test_joint_dataset)

    # target_key = []
    # for i in support_1_shots_novel_data:
    #     if i['label'] not in target_key:
    #         target_key.append(i['label'])
    # # 将sup_data中的目标领域的句子去掉
    # temp = []
    # for i in test_joint_data:
    #     if i['label'] not in target_key:
    #         temp.append(i)

    # write_jsonl_data(train_seen_data, trans_dataset + 'train_seen_data.jsonl')
    # write_jsonl_data(support_1_shots_novel_data, trans_dataset + 'support_5_shots_novel_data.jsonl')
    # write_jsonl_data(support_1_shots_joint_data, trans_dataset + 'support_5_shots_joint_data.jsonl')
    # write_jsonl_data(test_novel_data, trans_dataset + 'test_novel_data.jsonl')
    # write_jsonl_data(test_joint_data, trans_dataset + 'test_joint_data.jsonl')
    # with open(trans_dataset + 'train_seen_data.txt', 'w', encoding='utf-8') as fw:
    #     for line in train_seen_data:
    #         fw.write(line['sentence'])
    #         fw.write('\n')
    # with open(trans_dataset + 'valid.txt', 'w', encoding='utf-8') as fw:
    #     for line in temp:
    #         fw.write(line['sentence'])
    #         fw.write('\n')
    # print()
    print('saved')