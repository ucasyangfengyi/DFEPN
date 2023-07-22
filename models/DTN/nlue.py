import sys
sys.path.append('/opt/yfy/PythonProjects/DFEPN')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.DTN.bert_classifier import BertClassifier
from torch.utils.data import Dataset, DataLoader
import random
import collections
from collections import Counter
from utils.data import get_jsonl_data
import tqdm
import logging

from utils.python import set_seeds
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SEED = 3
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")

all_xg = []
key_dict = {}

class CNTALayerNorm(nn.Module):
    def __init__(self, embed_dim, variance_epsilon=1e-12):
        super(CNTALayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(embed_dim)).to(device)
        self.beta = nn.Parameter(torch.zeros(embed_dim)).to(device)
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class GeneratorNet(nn.Module):
    def __init__(self,num_labels=135):
        super(GeneratorNet, self).__init__()
        self.add_info = AddInfo()
        self.generator = Generator()
        self.layernorm = CNTALayerNorm(768)
        self.dropout = nn.Dropout(0.5)

    def forward(self, B1=None, B2=None, A=None, classifier=False):
        add_info = self.add_info(A, B1, B2)
        A_rebuild = self.generator(add_info)
        # A_rebuild = A + B1 - B2
        A_rebuild = A_rebuild + A
        A_rebuild = self.layernorm(A_rebuild)
        A_rebuild = self.dropout(A_rebuild)
        return A_rebuild


class AddInfo(nn.Module):
    def __init__(self):
        super(AddInfo, self).__init__()
        self.dense = nn.Linear(768, 1024)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.5)

    def forward(self, A, B1, B2):
        A = self.dense(A)
        A = self.activation(A)
        B1 = self.dense(B1)
        B1 = self.activation(B1)
        B2 = self.dense(B2)
        B2 = self.activation(B2)
        out = A+(B1-B2)
        # out = self.dropout(out)
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense = nn.Linear(1024, 768)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        out = self.dense(x)
        out = self.activation(out)
        # out = self.dropout(out)
        return out


def raw_data_to_labels_dict(data, shuffle=True):
    labels_dict = collections.defaultdict(list)
    for item in data:
        labels_dict[item['label']].append(item["sentence"])
    labels_dict = dict(labels_dict)
    if shuffle:
        for key, val in labels_dict.items():
            random.shuffle(val)
    return labels_dict

def create_xg(sup_data_dict,n_gen_support):
    xg = []
    if sup_data_dict != None:
        rand_keys = np.random.choice(list(sup_data_dict.keys()), n_gen_support, replace=False)
        for key, val in sup_data_dict.items():
            random.shuffle(val)
        xg = [[sup_data_dict[k][i] for i in range(2)] for k in rand_keys]
        # for i in xg:
        #     print(i)
        # print(xg)
    return xg,rand_keys

#从已有构建出的sup对中选择
# def create_xg(all_xg,n_gen_support):
#     xg = []
#     rand_keys = np.random.choice(len(all_xg), n_gen_support, replace=False)
#     random.shuffle(all_xg)
#     # xg = [[sup_data_dict[k][i] for i in range(2)] for k in rand_keys]
#     # for i in xg:
#     #     print(i)
#     # print(xg)
#     xg = [[all_xg[k][i] for i in range(2)] for k in rand_keys]
#     return xg





def generator_train(base_train_dataloader, train_data_dict,
                    bert_classifier,generator,
                    optimizer,criterion,output_path,
                    num_labels=150,log_every=10,n_gen_support=40,
                    ):
    count = 0
    all_xg = []
    with open(output_path+'/sup','r') as fr:
        for line in fr:
            all_xg.append(line.strip().split('\t'))
    for j in range(1):
        for data in tqdm.tqdm(base_train_dataloader):
            count += 1
            xg,rand_keys = create_xg(train_data_dict, n_gen_support)
            # xg = create_xg(all_xg,n_gen_support)
            # all_xg.append(xg)
            n_gen_support = len(xg)
            x = [item for xg_ in xg for item in xg_]
            bert_classifier.train()
            sentences = data['sentences']
            labels = data['labels']
            for i in labels:
                if i not in key_dict:
                    key_dict[i] = []
                    key_dict[i] += list(rand_keys)
                else:
                    key_dict[i] += list(rand_keys)
            # print(rand_keys)
            # print(labels)
            # print(labels)
            _, zs = bert_classifier(sentences)
            _, zg = bert_classifier(x)
            zg = zg.view(n_gen_support, 2, 768)
            zg_1 = zg[:, 0, :]
            zg_2 = zg[:, 1, :]
            for i in range(len(sentences)):
                generator.train()
                a = zs[i]
                # label = labels[i]
                # temp = [label] * n_gen_support
                # gen_labels = torch.stack(temp)
                gen_feature = generator(zg_1, zg_2, a)
                # probs_1 = bert_classifier.classifier(gen_feature)
                # predict_1 = torch.max(probs_1.view(-1,num_labels), 1)[1]
                # loss_1 = criterion(probs_1.view(-1, num_labels), gen_labels.view(-1))
                # correct_1 = torch.eq(predict_1, gen_labels.view(-1)).sum().float().item()
                #
                # temp = [label]
                # raw_label = torch.stack(temp)
                # probs_2 = bert_classifier.classifier(a)
                # predict_2 = torch.max(probs_2.view(-1,num_labels), 1)[1]
                # loss_2 = criterion(probs_2.view(-1, num_labels), raw_label.view(-1))
                # correct_2 = torch.eq(predict_2, gen_labels.view(-1)).sum().float().item()
                # acc = (correct_1 + correct_2) / (len(gen_labels.view(-1)) + 1)
                # # loss为generator生成的向量标签的损失和原来的向量的损失之和
                # loss = 0.2 * loss_1 + 0.8 * loss_2
                label = labels[i]
                temp = [label]
                gen_labels = torch.stack(temp)
                gen_labels = gen_labels.to(device)
                features = torch.cat((gen_feature, a.unsqueeze(0)), 0)
                features = torch.mean(features, 0)
                probs = bert_classifier.classifier(features)
                predict = torch.max(probs.view(-1,num_labels), 1)[1]
                correct = torch.eq(predict, gen_labels.view(-1)).sum().float().item()
                acc = correct / len(gen_labels.view(-1))
                loss = criterion(probs.view(-1, num_labels), gen_labels.view(-1))
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                if count % log_every == 0:
                    logger.info(f"generator train | loss: {loss:.4f} | acc: {acc:.4f}")
            # if count % 100 == 0:
            #     torch.save(bert_classifier.state_dict(),output_path + '/nlue4_classifier.pkl')
            #     torch.save(generator.state_dict(), output_path + '/nlue4_generator.pkl')
        torch.save(bert_classifier.state_dict(), output_path + '/nlue1_classifier.pkl' )
        torch.save(generator.state_dict(), output_path + '/nlue1_generator.pkl')


def generator_test(base_dev_dataloader, train_data_dict,
                    bert_classifier,generator,criterion,
                    num_labels=150,log_every=10,n_gen_support=40,
                    ):
    correct_num = 0
    test_num = 0
    with open(output_path+'/sup','r') as fr:
        for line in fr:
            all_xg.append(line.strip().split('\t'))
    for data in tqdm.tqdm(base_dev_dataloader):
        # xg = create_xg(train_data_dict, n_gen_support)
        xg = create_xg(all_xg, n_gen_support)

        n_gen_support = len(xg)
        x = [item for xg_ in xg for item in xg_]
        bert_classifier.eval()
        sentences = data['sentences']
        labels = data['labels'].to(device)
        _, zs = bert_classifier(sentences)
        _, zg = bert_classifier(x)
        zg = zg.view(n_gen_support, 2, 768)
        zg_1 = zg[:, 0, :]
        zg_2 = zg[:, 1, :]
        for i in range(len(sentences)):
            test_num += 1
            generator.eval()
            a = zs[i]
            label = labels[i]
            temp = [label]
            gen_labels = torch.stack(temp)
            gen_feature = generator(zg_1, zg_2, a)
            features = gen_feature
            features = torch.mean(features,0)
            # features = a
            features = 0.2 * features + 0.8 * a

            probs = bert_classifier.classifier(features)
            probs = probs.view(-1,num_labels)
            # S-N时计算acc
            temp = [torch.tensor(-100)] * 48
            temp = torch.stack(temp)
            probs[:, :48] = temp

            predict = torch.max(probs.view(-1, num_labels), 1)[1]
            correct_num += torch.eq(predict, gen_labels.view(-1)).sum().float().item()
            loss = criterion(probs.view(-1, num_labels), gen_labels.view(-1))
    acc = correct_num / test_num
    print('acc:',acc,'correct_num:',correct_num,'test_num:',test_num)


class BaseTrainDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, train_seen_data_path,support_1_shots_novel_data_path,n_classes,n_support):
        self.train_seen_data = get_jsonl_data(train_seen_data_path)
        self.support_data = get_jsonl_data(support_1_shots_novel_data_path)
        self.n_classes,self.n_support = n_classes,n_support
        self.label2id , self.id2label,self.sentence_list,self.label_list = self.map_label_sentence()
    def __len__(self):
        return len(self.sentence_list)
    def __getitem__(self, idx):
        sentence = self.sentence_list[idx]
        label = self.label_list[idx]
        sample = {'sentences': sentence, 'labels': label}
        return sample
    # 返回label与对应的id之间关系表
    def map_label_sentence(self):
        label2id = {}
        id2label = {}
        sentence_list = []
        label_list = []
        id = 0
        for data in self.train_seen_data:
            sentence_label = data['label']
            if sentence_label not in label2id:
                label2id[sentence_label] = id
                id2label[id] = sentence_label
                id += 1
            sentence = data['sentence']
            label = label2id[sentence_label]
            # label = sentence_label
            sentence_list.append(sentence)
            label_list.append(label)

        # support_label = []
        # for data in self.support_data:
        #     sentence_label = data['label']
        #     if sentence_label not in support_label:
        #         support_label.append(sentence_label)
        # rand_id = np.random.choice(support_label,self.n_classes,replace=False)
        # # rand_id = support_label
        # for i in rand_id:
        #     label2id[i] = id
        #     id2label[id] = i
        #     id += 1
        # support_sentences = []
        # support_labels = []
        # for data in self.support_data:
        #     sentence_label = data['label']
        #     sentence = data['sentence']
        #     if sentence_label in rand_id:
        #         # label = label2id[sentence_label]
        #         label = sentence_label
        #         support_sentences.append(sentence)
        #         support_labels.append(label)
        # sentence_list += support_sentences * int(100 / self.n_support)
        # label_list += support_labels * int(100 / self.n_support)
        return label2id,id2label,sentence_list,label_list

class BaseDevDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, train_jsonl,label2id):
        self.train_data = get_jsonl_data(train_jsonl)
        self.label2id = label2id
        self.sentence_list,self.label_list = self.map_label_sentence()
    def __len__(self):
        return len(self.sentence_list)
    def __getitem__(self, idx):
        sentence = self.sentence_list[idx]
        label = self.label_list[idx]
        sample = {'sentences': sentence, 'labels': label}
        return sample
    # 返回label与对应的id之间关系表
    def map_label_sentence(self):
        sentence_list = []
        label_list = []
        for data in self.train_data:
            sentence_label = data['label']
            sentence = data['sentence']
            if sentence_label in self.label2id:
                label = self.label2id[sentence_label]
                sentence_list.append(sentence)
                label_list.append(label)
        return sentence_list,label_list


if __name__ == '__main__':
    log_every = 10
    epochs = 5
    seed = 5
    set_seeds(seed)
    output_path = 'save'
    # bert_path = '/opt/yfy/PythonProjects/Fewshot-SLU/data/bert/bert-base-uncased/'
    bert_path = '/opt/yfy/PythonProjects/DFEPN/language_modeling/transformer_models/nlue/kfold_1/fine-tuned/'
    trans_dataset = '/opt/yfy/PythonProjects/DFEPN/dataset/trans_dataset/NLUE/KFold_1/'

    train_seen_data_path = trans_dataset + 'auxiliary_samples.jsonl'
    support_1_shots_novel_data_path = trans_dataset + 'support_1_shots_novel_data.jsonl'
    test_novel_data_path = trans_dataset + 'test_novel_data.jsonl'
    test_joint_data_path = trans_dataset + 'test_joint_data.jsonl'

    n_support = 1
    n_classes = 16
    n_gen_support = 20
    num_labels = 20

    # 数据集
    train_support_data = BaseTrainDataset(train_seen_data_path,support_1_shots_novel_data_path,n_classes,n_support)
    label2id = train_support_data.label2id
    train_support_dataloader = DataLoader(train_support_data, batch_size=16, shuffle=True)

    test_novel_data = BaseDevDataset(test_novel_data_path,label2id)
    test_novel_dataloader = DataLoader(test_novel_data, batch_size=16, shuffle=False)

    test_joint_data = BaseDevDataset(test_joint_data_path, label2id)
    test_joint_dataloader = DataLoader(test_joint_data, batch_size=16, shuffle=False)

    train_data = []
    # for sentence,label in zip(train_support_data.sentence_list,train_support_data.label_list):
    #     train_data.append({'sentence':sentence,'label':label})
    train_data = get_jsonl_data(train_seen_data_path)
    train_data_dict = raw_data_to_labels_dict(train_data, shuffle=True)
    criterion = nn.CrossEntropyLoss().cuda()

    # 模型训练
    bert_classifier = BertClassifier(bert_path, num_labels).to(device)
    # bert_classifier.load_state_dict(torch.load('save/bert_classifier_15_finetune.pkl'))
    generator = GeneratorNet().to(device)
    # generator.load_state_dict(torch.load('save/generator.pkl.0'))
    # par = list(generator.named_parameters())
    # print(par[0])
    optimizer = torch.optim.Adam([
        {'params': bert_classifier.bert.parameters(), 'lr': 2e-5},
        {'params': bert_classifier.classifier.parameters(), 'lr': 1e-4},
        {'params': generator.parameters(), 'lr': 1e-4}
    ]
    )
    generator_train(train_support_dataloader, train_data_dict, bert_classifier, generator,
                    optimizer, criterion, output_path,
                    num_labels, log_every,n_gen_support)
    # for key in key_dict:
    #     a = Counter(key_dict[key])
    #     key_dict[key] = a
    # print(key_dict)
    # with open(output_path + '/sup','w',encoding='utf-8') as fw:
    #     for line in all_xg:
    #         for pair in line:
    #             fw.write(pair[0] + '\t' + pair[1] + '\n')

    # 测试模型
    # best模型在训练300次左右出现
    # bert_classifier = BertClassifier(bert_path, num_labels).to(device)
    # bert_classifier.load_state_dict(torch.load('save/nlue5_classifier.pkl'))
    # generator = GeneratorNet().to(device)
    # generator.load_state_dict(torch.load('save/nlue5_generator.pkl'))
    # generator_test(test_novel_dataloader, train_data_dict,
    #                bert_classifier, generator, criterion,
    #                num_labels, log_every, n_gen_support,
    #                )









