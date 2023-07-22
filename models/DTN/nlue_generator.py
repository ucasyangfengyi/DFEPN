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

class GeneratorNet(nn.Module):
    def __init__(self,num_labels=135):
        super(GeneratorNet, self).__init__()
        self.add_info = AddInfo()
        self.generator = Generator()
        # self.b1 = nn.Parameter(torch.FloatTensor([0.5]))

    def forward(self, B1=None, B2=None, A=None, classifier=False):
        add_info = self.add_info(A, B1, B2)
        A_rebuild = self.generator(add_info)
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
        out = self.dropout(out)
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
        out = self.dropout(out)
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

# def create_xg(sup_data_dict,n_gen_support):
#     xg = []
#     if sup_data_dict != None:
#         rand_keys = np.random.choice(list(sup_data_dict.keys()), n_gen_support, replace=True)
#         # for key, val in sup_data_dict.items():
#         #     random.shuffle(val)
#         for key in rand_keys:
#             random.shuffle(sup_data_dict[key])
#             xg.append([sup_data_dict[key][i] for i in range(2)])
#     return xg

def create_xg(sup_data_dict,n_gen_support,label_dict,label):
    xg = []
    for i in range(n_gen_support):
        # rand_keys = np.random.choice(list(sup_data_dict.keys()), n_gen_support, replace=False)
        rand_keys = label_dict[label.item()]
        for key, val in sup_data_dict.items():
            random.shuffle(val)
        # xg.append(sup_data_dict[rand_keys][0])
        xg.append([sup_data_dict[rand_keys][0],sup_data_dict[rand_keys][1]])
    return xg

def create_xg_test(sup_data_dict,n_gen_support):
    xg = []
    # rand_keys = np.random.choice(list(sup_data_dict.keys()), 1, replace=False)
    # for i in range(n_gen_support):
    #     for key, val in sup_data_dict.items():
    #         random.shuffle(val)
    #     xg.append([sup_data_dict[rand_keys[0]][0],sup_data_dict[rand_keys[0]][1]])
    if sup_data_dict != None:
        rand_keys = np.random.choice(list(sup_data_dict.keys()), n_gen_support, replace=False)
        for key, val in sup_data_dict.items():
            random.shuffle(val)
        xg = [[sup_data_dict[k][i] for i in range(2)] for k in rand_keys]
    return xg


def generator_train(base_train_dataloader, train_data_dict,label_dict,
                    net,isfinetune,n_support,nlue_id,
                    bert_classifier,generator,
                    optimizer,criterion,output_path,
                    num_labels,log_every,n_gen_support,
                    test_novel_dataloader,test_joint_dataloader,test_novel_dict,test_joint_dict,
                    mask_label,new_label,label2id,num_episodes
                    ):
    count = 0
    for j in range(1):
        for data in tqdm.tqdm(base_train_dataloader):
            count += 1
            bert_classifier.train()
            sentences = data['sentences']
            labels = data['labels'].to(device)
            _, zs = bert_classifier(sentences)

            xg = create_xg_test(train_data_dict, n_gen_support)
            n_gen_support = len(xg)
            x = [item for xg_ in xg for item in xg_]
            _, zg = bert_classifier(x)
            zg = zg.view(n_gen_support, 2, 768)
            zg_1 = zg[:, 0, :]
            zg_2 = zg[:, 1, :]
            for i in range(len(sentences)):
                generator.train()
                a = zs[i]
                label = labels[i]
                temp = [label]
                gen_labels = torch.stack(temp)
                # xg = create_xg_test(train_data_dict, n_gen_support,label_dict,label)
                # n_gen_support = len(xg)
                # x = [item for xg_ in xg for item in xg_]
                # _, zg = bert_classifier(x)
                # zg = zg.view(n_gen_support, 2, 768)
                # zg_1 = zg[:, 0, :]
                # zg_2 = zg[:, 1, :]

                if net == 'mynet':
                    gen_feature = generator(zg_1, zg_2, a)
                    gen_feature = torch.cat((gen_feature,a.unsqueeze(0)),0)
                    features = torch.mean(gen_feature, 0)
                    # features = 0.5 * features + 0.5 * a
                    # print(generator.b1)
                elif net == 'bert':
                    features = a
                    # print(features)
                probs = bert_classifier.classifier(features)
                predict = torch.max(probs.view(-1,num_labels), 1)[1]
                correct = torch.eq(predict, gen_labels.view(-1)).sum().float().item()
                acc = correct / len(gen_labels.view(-1))
                loss = criterion(probs.view(-1, num_labels), gen_labels.view(-1))
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            # if count % log_every == 0:
            #     logger.info(f"generator train | loss: {loss:.4f} | acc: {acc:.4f}")
            # if count % 100 == 0:
            #     # break
            #     print('第',str(count),'轮实验结果')
            #     noneps_test(test_novel_dataloader, train_data_dict, mask_label,
            #                 bert_classifier, generator, criterion,
            #                 num_labels, n_gen_support, 'novel', net)
            #
            #     noneps_test(test_joint_dataloader, train_data_dict, mask_label,
            #                 bert_classifier, generator, criterion,
            #                 num_labels, n_gen_support, 'joint', net)

                # eps_test(test_novel_dict, train_data_dict, new_label, label2id, 'novel', net,
                #          bert_classifier, generator, criterion,
                #          num_labels, n_gen_support, num_episodes)
                #
                # eps_test(test_joint_dict, train_data_dict, new_label, label2id, 'joint', net,
                #          bert_classifier, generator, criterion,
                #          num_labels, n_gen_support, num_episodes)

        # if isfinetune == True:
        #     torch.save(bert_classifier.state_dict(), output_path +
        #                '/'+net+'.nlue_'+str(nlue_id)+'.'+str(n_support)+'_shot.classifier.finetune.pkl')
        #     torch.save(generator.state_dict(), output_path +
        #                '/'+net+'.nlue_'+str(nlue_id)+'.'+str(n_support)+'_shot.generator.finetune.pkl')
        #     # break
        # if isfinetune == False:
        #     # 使用base不适合提前结束，因此直接进行一轮完整训练
        #     torch.save(bert_classifier.state_dict(), output_path +
        #                '/' + net + '.nlue_' + str(nlue_id) + '.' + str(n_support) + '_shot.classifier.base.pkl')
        #     torch.save(generator.state_dict(), output_path +
        #                '/' + net + '.nlue_' + str(nlue_id) + '.' + str(n_support) + '_shot.generator.base.pkl')




def noneps_test(base_dev_dataloader, train_data_dict,mask_label,
                    bert_classifier,generator,criterion,
                    num_labels,n_gen_support,tgt,net
                    ):
    correct_num = 0
    test_num = 0
    for data in tqdm.tqdm(base_dev_dataloader):
        bert_classifier.eval()
        sentences = data['sentences']
        labels = data['labels'].to(device)
        _, zs = bert_classifier(sentences)
        xg = create_xg_test(train_data_dict, n_gen_support)
        n_gen_support = len(xg)
        x = [item for xg_ in xg for item in xg_]
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
            if net == 'mynet':
                gen_feature = generator(zg_1, zg_2, a)
                # gen_feature = torch.cat((gen_feature, a.unsqueeze(0)), 0)
                features = gen_feature
                # features = torch.mean(gen_feature, 0)
                features = torch.mean(features, 0)
                features = 0.2 * features + 0.8 * a
                # features = generator.b1 * features + (1 - generator.b1) * a
            elif net == 'bert':
                features = a

            probs = bert_classifier.classifier(features)
            probs = probs.view(-1,num_labels)
            # S-N时计算acc
            if tgt=='novel':
                # temp = torch.tensor(-100)
                # temp = [temp]*48
                # temp = torch.stack(temp)
                # probs[:,:48] = temp

                probs = probs + 1000
                probs = torch.mul(probs,mask_label)
                probs = probs - 1000
            # print(probs.max())
            predict = torch.max(probs.view(-1, num_labels), 1)[1]
            correct_num += torch.eq(predict, gen_labels.view(-1)).sum().float().item()
            loss = criterion(probs.view(-1, num_labels), gen_labels.view(-1))
    acc = correct_num / test_num
    print('noneps test ******','net:',net,'tgt',tgt,'acc:',acc,'correct_num:',correct_num,'test_num:',test_num)
    return acc

def eps_test(test_dict, train_data_dict,new_label,label2id,tgt,net,
                    bert_classifier,generator,criterion,
                    num_labels,n_gen_support,num_episodes,
                    ):
    correct_num = 0
    test_num = 0
    for episode in tqdm.tqdm(range(num_episodes)):
        if tgt == 'novel':
            rand_label = np.random.choice(new_label,5,replace=False)
        elif tgt == 'joint':
            rand_label = np.random.choice(list(label2id.keys()),10,replace=False)
        mask_label = torch.zeros(num_labels)
        for i in rand_label:
            mask_label[label2id[i]] = 1
        mask_label = mask_label.to(device)
        # 将测试的句子乱序
        for key, val in test_dict.items():
            random.shuffle(val)
        for key in rand_label:
            sentences = test_dict[key][0:5]
            labels = torch.tensor([label2id[key]] * 5).to(device)
            xg = create_xg_test(train_data_dict, n_gen_support)
            n_gen_support = len(xg)
            x = [item for xg_ in xg for item in xg_]
            bert_classifier.eval()
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

                if net == 'mynet':
                    features = 0.2 * features + 0.8 * a
                    # features = a
                elif net == 'bert':
                    features = a

                probs = bert_classifier.classifier(features)
                probs = probs.view(-1,num_labels)
                probs = probs + 1000
                probs = torch.mul(probs,mask_label)
                probs = probs - 1000
                predict = torch.max(probs.view(-1, num_labels), 1)[1]
                correct_num += torch.eq(predict, gen_labels.view(-1)).sum().float().item()
                loss = criterion(probs.view(-1, num_labels), gen_labels.view(-1))
    acc = correct_num / test_num
    print('eps test ******','net:',net,'tgt',tgt,'acc:',acc,'correct_num:',correct_num,'test_num:',test_num)
    return acc
class BaseTrainDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self,train_data_dict, train_seen_data_path,support_1_shots_novel_data_path,n_classes,n_support,label2id,new_label):
        self.train_seen_data = get_jsonl_data(train_seen_data_path)
        self.train_data_dict = train_data_dict
        self.support_data = get_jsonl_data(support_1_shots_novel_data_path)
        self.n_classes,self.n_support,self.label2id,self.new_label = n_classes,n_support,label2id,new_label
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
        for data in self.train_seen_data:
            sentence_label = data['label']
            sentence = data['sentence']
            label = self.label2id[sentence_label]
            sentence_list.append(sentence)
            label_list.append(label)
        #
        # for key in self.train_data_dict:
        #     if len(self.train_data_dict[key]) > 100:
        #         sentence_list += self.train_data_dict[key]
        #         label_list += [self.label2id[key]] * len(self.train_data_dict[key])
        #     else:
        #         # a = int(200 / len(self.train_data_dict[key]))
        #         sentence_list += self.train_data_dict[key] * int(150 / len(self.train_data_dict[key]))
        #         label_list += [self.label2id[key]] * len(self.train_data_dict[key]) * int(150 / len(self.train_data_dict[key]))
        # rand_id = np.random.choice(self.new_label,self.n_classes,replace=False)
        rand_id = self.new_label
        support_sentences = []
        support_labels = []
        for data in self.support_data:
            sentence_label = data['label']
            sentence = data['sentence']
            if sentence_label in rand_id:
                label = self.label2id[sentence_label]
                support_sentences.append(sentence)
                support_labels.append(label)
        sentence_list += support_sentences * int(100 / self.n_support)
        label_list += support_labels * int(100 / self.n_support)
        return sentence_list,label_list

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


# 原来写的生成label和id对应表的方法，用来加载原模型
def raw_label2id(train_seen_data_path, support_n_shots_novel_data_path):
    label2id = {}
    id2label = {}
    new_label = []
    train_seen_data = get_jsonl_data(train_seen_data_path)
    support_data = get_jsonl_data(support_n_shots_novel_data_path)
    id = 0
    for data in train_seen_data:
        if data['label'] not in label2id:
            label2id[data['label']] = id
            id2label[id] = data['label']
            id += 1
    for data in support_data:
        if data['label'] not in label2id:
            new_label.append(data['label'])
            label2id[data['label']] = id
            id2label[id] = data['label']
            id += 1
    return label2id,id2label,new_label


def Manager():
    return


if __name__ == '__main__':
    log_every = 20
    epochs = 1
    n_support = 1
    n_classes = 16
    n_gen_support = 40
    isfinetune = True  # True or False
    istrain = False
    ispredict = True
    num_episodes = 1000
    net = 'bert'  # mynet or bert
    num_labels = 48 + n_classes


    output_path = 'save'
    all_label_path = '/opt/yfy/PythonProjects/DFEPN/dataset/raw_dataset/NLUE/label.tsv'
    new_label_path = '/opt/yfy/PythonProjects/DFEPN/dataset/raw_dataset/NLUE/new_intents'
    label_dict_path = '/opt/yfy/PythonProjects/DFEPN/dataset/raw_dataset/NLUE/label_dict'
    label2id = {}
    id2label = {}
    new_label = []
    id = 0
    with open(all_label_path, 'r') as fr:
        for line in fr:
            label2id[line.strip()] = id
            id2label[id] = line.strip()
            id += 1
    with open(new_label_path, 'r') as fr:
        for line in fr:
            new_label.append(line.strip())
    mask_label = torch.zeros(num_labels)
    for i in new_label:
        mask_label[label2id[i]] = 1
    mask_label = mask_label.to(device)

    label_dict = {}
    with open(label_dict_path,'r') as fr:
        for line in fr:
            temp = line.strip().split('#')
            a = label2id[temp[0]]
            b = temp[1]
            label_dict[a] = b
    noneps_sj = []
    noneps_sn = []
    eps_sj = []
    eps_sn = []


    nlue_id = 1
    # seed = 2
    best_noneps_h_acc = 0
    # for nlue_id in range(8,9):
    for seed in range(10):
        set_seeds(seed)
        print('nlue_id:',nlue_id,'   ****   seed:',seed)
        bert_path_base = '/opt/yfy/PythonProjects/Fewshot-SLU/data/bert/bert-base-uncased/'
        bert_path_finetune = '/opt/yfy/PythonProjects/DFEPN/language_modeling/transformer_models/nlue/kfold_'\
                             +str(nlue_id)+'/fine-tuned/'
        trans_dataset = '/opt/yfy/PythonProjects/DFEPN/dataset/trans_dataset/NLUE/KFold_'+str(nlue_id)+'/'

        train_seen_data_path = trans_dataset + 'train_seen_data.jsonl'
        support_n_shots_novel_data_path = trans_dataset + 'support_' + str(n_support) + '_shots_novel_data.jsonl'
        test_novel_data_path = trans_dataset + 'test_novel_data.jsonl'
        test_joint_data_path = trans_dataset + 'test_joint_data.jsonl'

        # label2id, id2label, new_label = raw_label2id(train_seen_data_path, support_n_shots_novel_data_path)
        # mask_label = torch.zeros(num_labels)
        # for i in new_label:
        #     mask_label[label2id[i]] = 1
        # mask_label = mask_label.to(device)

        # 辅助样例数据集，用来生成样例对，就是train_seen_data数据集
        train_data = get_jsonl_data(train_seen_data_path)
        train_data_dict = raw_data_to_labels_dict(train_data, shuffle=True)
        # len(train_data_dict[i])

        # 数据集
        train_support_data = BaseTrainDataset(train_data_dict,train_seen_data_path, support_n_shots_novel_data_path, n_classes, n_support, label2id, new_label)
        train_support_dataloader = DataLoader(train_support_data, batch_size=16, shuffle=True)

        test_novel_data = BaseDevDataset(test_novel_data_path,label2id)
        test_novel_dataloader = DataLoader(test_novel_data, batch_size=16, shuffle=False)
        test_novel_dict = raw_data_to_labels_dict(get_jsonl_data(test_novel_data_path), shuffle=True)

        test_joint_data = BaseDevDataset(test_joint_data_path, label2id)
        test_joint_dataloader = DataLoader(test_joint_data, batch_size=16, shuffle=False)
        test_joint_dict = raw_data_to_labels_dict(get_jsonl_data(test_joint_data_path), shuffle=True)


        # for i in train_data_dict:
        #     print(len(train_data_dict[i]))
        # break
        # 模型初始化
        criterion = nn.CrossEntropyLoss().cuda()
        if isfinetune == True:
            bert_classifier = BertClassifier(bert_path_finetune, num_labels).to(device)
        else:
            bert_classifier = BertClassifier(bert_path_base, num_labels).to(device)
        generator = GeneratorNet().to(device)
        # par = list(generator.named_parameters())
        # print(par[0])

        optimizer = torch.optim.Adam([
            {'params': bert_classifier.bert.parameters(), 'lr': 2e-5},
            {'params': bert_classifier.classifier.parameters(), 'lr': 1e-4},
            {'params': generator.parameters(), 'lr': 1e-4}
        ]
        )
        # 模型训练
        if istrain:
            # if net == 'mynet':
            #     bert_classifier.load_state_dict(torch.load(output_path +
            #         '/' + 'bert' + '.nlue_' + str(nlue_id) + '.' + str(n_support) + '_shot.classifier.finetune.pkl'))
            #     generator.load_state_dict(torch.load(output_path +
            #         '/' + 'bert' + '.nlue_' + str(nlue_id) + '.' + str(n_support) + '_shot.generator.finetune.pkl'))
            # # 冻结特征提取的网络
            # # for param in bert_classifier.parameters():
            # #     param.requires_grad = False

            # 加载已经训练一轮的模型，再训练一轮看看结果
            # bert_classifier.load_state_dict(torch.load(output_path +
            #                                            '/' + net + '.nlue_' + str(nlue_id) + '.' + str(
            #     n_support) + '_shot.classifier.finetune.pkl'))
            # generator.load_state_dict(torch.load(output_path +
            #                                      '/' + net + '.nlue_' + str(nlue_id) + '.' + str(
            #     n_support) + '_shot.generator.finetune.pkl'))

            generator_train(train_support_dataloader, train_data_dict,label_dict,
                        net,isfinetune,n_support,nlue_id,
                        bert_classifier,generator,
                        optimizer,criterion,output_path,
                        num_labels,log_every,n_gen_support,
                        test_novel_dataloader, test_joint_dataloader, test_novel_dict, test_joint_dict,
                        mask_label, new_label, label2id, num_episodes
                        )

        # 测试模型
        # best模型在训练300次左右出现
        if ispredict:
            if isfinetune:
                bert_classifier.load_state_dict(torch.load(output_path +
                                   '/'+net+'.nlue_'+str(nlue_id)+'.'+str(n_support)+'_shot.classifier.finetune.pkl'))
                generator.load_state_dict(torch.load(output_path +
                                   '/'+net+'.nlue_'+str(nlue_id)+'.'+str(n_support)+'_shot.generator.finetune.pkl'))
            else:
                bert_classifier.load_state_dict(torch.load(output_path +
                                '/' + net + '.nlue_' + str(nlue_id) + '.' + str(n_support) + '_shot.classifier.base.pkl'))
                generator.load_state_dict(torch.load(output_path +
                                '/' + net + '.nlue_' + str(nlue_id) + '.' + str(n_support) + '_shot.generator.base.pkl'))

        # bert_classifier.load_state_dict((torch.load('temp_save/best-16-classifier')))
        # generator.load_state_dict(torch.load('temp_save/best-16-generator'))

        noneps_sn_acc = noneps_test(test_novel_dataloader, train_data_dict, mask_label,
                    bert_classifier, generator, criterion,
                    num_labels, n_gen_support, 'novel', net)
        noneps_sn.append(noneps_sn_acc)

        noneps_sj_acc = noneps_test(test_joint_dataloader, train_data_dict, mask_label,
                    bert_classifier, generator, criterion,
                    num_labels, n_gen_support, 'joint', net)
        noneps_sj.append(noneps_sj_acc)
        noneps_h_acc = 2.0 / (1.0 / noneps_sn_acc + 1.0 / noneps_sj_acc)
        print('noneps_h_acc:',noneps_h_acc)

        if istrain:
            if noneps_h_acc > best_noneps_h_acc:
                best_noneps_h_acc = noneps_h_acc
                print('best_noneps_h_acc:',best_noneps_h_acc)
                if isfinetune == True:
                    torch.save(bert_classifier.state_dict(), output_path +
                               '/'+net+'.nlue_'+str(nlue_id)+'.'+str(n_support)+'_shot.classifier.finetune.pkl')
                    torch.save(generator.state_dict(), output_path +
                               '/'+net+'.nlue_'+str(nlue_id)+'.'+str(n_support)+'_shot.generator.finetune.pkl')
                    # break
                if isfinetune == False:
                    # 使用base不适合提前结束，因此直接进行一轮完整训练
                    torch.save(bert_classifier.state_dict(), output_path +
                               '/' + net + '.nlue_' + str(nlue_id) + '.' + str(n_support) + '_shot.classifier.base.pkl')
                    torch.save(generator.state_dict(), output_path +
                               '/' + net + '.nlue_' + str(nlue_id) + '.' + str(n_support) + '_shot.generator.base.pkl')


        # eps_sn_acc = eps_test(test_novel_dict, train_data_dict, new_label, label2id, 'novel', net,
        #          bert_classifier, generator, criterion,
        #          num_labels, n_gen_support, num_episodes)
        # eps_sn.append(eps_sn_acc)
        # eps_sj_acc = eps_test(test_joint_dict, train_data_dict, new_label, label2id, 'joint', net,
        #          bert_classifier, generator, criterion,
        #          num_labels, n_gen_support, num_episodes)
        # eps_sj.append(eps_sj_acc)
        # eps_h_acc = 2.0 / (1.0 / eps_sn_acc + 1.0 / eps_sj_acc)



    noneps_sn = np.array(noneps_sn)
    noneps_sj = np.array(noneps_sj)
    eps_sn = np.array(eps_sn)
    eps_sj = np.array(eps_sj)
    # print('noneps_sn:',noneps_sn)
    # print('noneps_sj:',noneps_sj)
    # print('eps_sn:',eps_sn)
    # print('eps_sj:',eps_sj)
    noneps_sj_avg = np.mean(noneps_sj)
    noneps_sn_avg = np.mean(noneps_sn)
    eps_sj_avg = np.mean(eps_sj)
    eps_sn_avg = np.mean(eps_sn)
    print('noneps_sj_avg:',noneps_sj_avg)
    print('noneps_sn_avg:',noneps_sn_avg)
    print('eps_sj_avg:',eps_sj_avg)
    print('eps_sn_avg:',eps_sn_avg)
    noneps_h_acc_avg = 2.0 / (1.0 / noneps_sn_avg + 1.0 / noneps_sj_avg)
    eps_h_acc_avg = 2.0 / (1.0 / eps_sn_avg + 1.0 / eps_sj_avg)
    print('noneps_h_acc_avg:',noneps_h_acc_avg)
    print('eps_h_acc_avg:',eps_h_acc_avg)











