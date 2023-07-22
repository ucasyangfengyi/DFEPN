import sys
sys.path.append('/opt/yfy/PythonProjects/DFEPN')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.DTN.bert_classifier import BertClassifier,BaseTrainDataset,BaseDevDataset
from torch.utils.data import Dataset, DataLoader
import random
import collections
from utils.data import get_jsonl_data
import tqdm
import logging
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

def create_xg(sup_data_dict,n_gen_support):
    xg = []
    if sup_data_dict != None:
        rand_keys = np.random.choice(list(sup_data_dict.keys()), n_gen_support, replace=False)
        for key, val in sup_data_dict.items():
            random.shuffle(val)
        xg = [[sup_data_dict[k][i] for i in range(2)] for k in rand_keys]
    return xg


# def generator_train(base_train_dataloader, train_data_dict,bert_classifier,generator,
#                     optimizer,criterion,output_path,
#                     num_labels=150,log_every=10,n_gen_support=40):
#     count = 0
#     for j in range(10):
#         for data in tqdm.tqdm(base_train_dataloader):
#             count += 1
#             xg = create_xg(train_data_dict, n_gen_support)
#             n_gen_support = len(xg)
#             x = [item for xg_ in xg for item in xg_]
#             bert_classifier.train()
#             # bert_classifier.eval()
#             sentences = data['sentences']
#             labels = data['labels'].to(device)
#             _, zs = bert_classifier(sentences)
#             _, zg = bert_classifier(x)
#
#             zg = zg.view(n_gen_support, 2, 768)
#             zg_1 = zg[:, 0, :]
#             zg_2 = zg[:, 1, :]
#
#             for i in range(len(sentences)):
#                 generator.train()
#                 # generator.eval()
#                 optimizer.zero_grad()
#                 a = zs[i]
#                 label = labels[i]
#                 temp = [label] * (n_gen_support + 1)
#                 gen_labels = torch.stack(temp)
#                 gen_feature = generator(zg_1, zg_2, a)
#                 features = torch.cat((gen_feature,a.unsqueeze(0)))
#                 logits = bert_classifier.classifier(features)
#                 loss = criterion(logits.view(-1, num_labels), gen_labels.view(-1))
#                 predict = torch.max(logits, 1)[1]
#                 correct = torch.eq(predict, gen_labels).sum().float().item()
#                 acc = correct / len(gen_labels)
#                 loss.backward(retain_graph=True)
#                 optimizer.step()
#                 if count % log_every == 0:
#                     logger.info(f"generator train | loss: {loss:.4f} | acc: {acc:.4f}")
#         torch.save(bert_classifier.state_dict(), output_path + '/nofin_bert_classifier.pkl.' + str(j))
#         torch.save(generator.state_dict(), output_path + '/nofin_generator.pkl.' + str(j))

def generator_train(base_train_dataloader, train_data_dict,
                    bert_classifier,generator,
                    optimizer,criterion,output_path,
                    num_labels=150,log_every=10,n_gen_support=40,
                    ):
    count = 0
    for j in range(10):
        for data in tqdm.tqdm(base_train_dataloader):
            count += 1
            xg = create_xg(train_data_dict, n_gen_support)
            n_gen_support = len(xg)
            x = [item for xg_ in xg for item in xg_]
            bert_classifier.train()
            sentences = data['sentences']
            labels = data['labels'].to(device)
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
                if label < 135:
                    features = a
                else:
                    print('训练目标类')
                    features = torch.mean(gen_feature,0)
                    features = 0.2 * features + 0.8 * a
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
            if count % 100 == 0:
                torch.save(bert_classifier.state_dict(),output_path + '/28temp_bert_classifier.pkl')
                torch.save(generator.state_dict(), output_path + '/28temp_generator.pkl')
        torch.save(bert_classifier.state_dict(), output_path + '/28_bert_classifier.pkl.' + str(j))
        torch.save(generator.state_dict(), output_path + '/28_generator.pkl.' + str(j))


def generator_test(base_dev_dataloader, train_data_dict,
                    bert_classifier,generator,criterion,
                    num_labels=150,log_every=10,n_gen_support=40,
                    ):
    xxx = 0
    count = 0
    probs_max = torch.tensor(100)
    for data in tqdm.tqdm(base_dev_dataloader):
        count += 1
        xg = create_xg(train_data_dict, n_gen_support)
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
            generator.eval()
            a = zs[i]
            label = labels[i]
            # temp = [label] * (n_gen_support + 1)
            # temp = [label] * n_gen_support
            temp = [label]
            gen_labels = torch.stack(temp)
            gen_feature = generator(zg_1, zg_2, a)
            features = gen_feature
            # features = torch.cat((gen_feature, a.unsqueeze(0)), 0)
            features = torch.mean(features,0)
            # features = a
            features = 0.2 * features + 0.8 * a
            # features = 0.2*features + 0.8*a
            probs = bert_classifier.classifier(features)
            probs = probs.view(-1,num_labels)
            # probs = F.normalize(probs)
            temp = [torch.tensor(-100)] * 135
            temp = torch.stack(temp)
            probs[:,:135] = temp
            all_probs = torch.tensor(0).to(device)
            for i in probs[:,135:].squeeze():
                all_probs = all_probs + i
            m = nn.Softmax(dim=1)
            probs[:,135:] = m(probs[:,135:])
            # if torch.max(probs) <= 0.3:
            #     predict = torch.tensor(150).to(device)
            #     xxx += 1
            # else:
            #     predict = torch.max(probs.view(-1,num_labels), 1)[1]
            predict = torch.max(probs.view(-1, num_labels), 1)[1]
            correct = torch.eq(predict, gen_labels.view(-1)).sum().float().item()
            acc = correct / len(gen_labels.view(-1))
            # if torch.max(probs) < probs_max:
            #     probs_max = torch.max(probs)
            # if acc == 1:
            #     xxx += 1
                # print(all_probs)
                # print(torch.max(probs))
            if torch.max(probs) <= 0.35:
                xxx += 1
            # if acc==1 and torch.max(probs) > 0.35:
            #     xxx += 1
            # print(all_probs)
    print(xxx)
    print(probs_max)
if __name__ == '__main__':
    num_labels = 150
    log_every = 10
    epochs = 10
    output_path = 'save'
    # bert_path = '/opt/yfy/PythonProjects/Fewshot-SLU/data/bert/bert-base-uncased/'
    bert_path = '/opt/yfy/PythonProjects/DFEPN/language_modeling/transformer_models/nlue_kfold_1/fine-tuned/'
    train_path = '../../dataset/trans_dataset/clinc150/banking/train_15_5.jsonl'
    target_test_path = '../../dataset/trans_dataset/clinc150/banking/target.test.jsonl'
    oos_test_path = '../../dataset/trans_dataset/clinc150/oos/oos.test.jsonl'
    n_support = 5
    n_classes = 15
    n_query = 10
    n_gen_support = 40

    # 数据集
    base_train_dataset = BaseTrainDataset(train_path)
    label2id = base_train_dataset.label2id
    base_train_dataloader = DataLoader(base_train_dataset, batch_size=15, shuffle=True)
    target_test_dataset = BaseDevDataset(target_test_path,label2id)
    target_test_dataloader = DataLoader(target_test_dataset, batch_size=15, shuffle=False)

    label2id['oos'] = torch.tensor(150)
    oos_test_dataset = BaseDevDataset(oos_test_path, label2id)
    oos_test_dataloader = DataLoader(oos_test_dataset, batch_size=15, shuffle=False)

    train_data = get_jsonl_data(train_path)
    train_data_dict = raw_data_to_labels_dict(train_data, shuffle=True)
    criterion = nn.CrossEntropyLoss().cuda()
    #
    # # 模型训练
    # bert_classifier = BertClassifier(bert_path, num_labels).to(device)
    # # bert_classifier.load_state_dict(torch.load('save/bert_classifier_15_finetune.pkl'))
    # generator = GeneratorNet().to(device)
    # # generator.load_state_dict(torch.load('save/generator.pkl.0'))
    # optimizer = torch.optim.Adam([
    #     {'params': bert_classifier.bert.parameters(), 'lr': 2e-5},
    #     {'params': bert_classifier.classifier.parameters(), 'lr': 1e-4},
    #     {'params': generator.parameters(), 'lr': 1e-4}
    # ]
    # )
    # generator_train(base_train_dataloader, train_data_dict, bert_classifier, generator,
    #                 optimizer, criterion, output_path,
    #                 num_labels, log_every,n_gen_support)

    # 测试模型
    # best模型在训练300次左右出现
    bert_classifier = BertClassifier(bert_path, num_labels).to(device)
    bert_classifier.load_state_dict(torch.load('save/new_best_classifier.pkl'))
    generator = GeneratorNet().to(device)
    generator.load_state_dict(torch.load('save/new_best_generator.pkl'))
    generator_test(oos_test_dataloader, train_data_dict,
                   bert_classifier, generator, criterion,
                   num_labels, log_every, n_gen_support,
                   )









