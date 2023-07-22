import sys
sys.path.append('/opt/yfy/PythonProjects/DFEPN')

import json
import argparse
# from models.encoders.bert_encoder import BERTEncoder
from models.DTN.bert_classifier import BertClassifier
from models.DTN.nlue import GeneratorNet
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
from utils.few_shot import create_episode,create_snips_episode
from utils.my_math import euclidean_dist, cosine_similarity
# from models.generator.generator import GeneratorNet
import tqdm
from copy import deepcopy
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pylab as plt
from matplotlib import cm


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.simplefilter('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
kfold = '10'
raw_gen_support = 25
# save_file = '/1_shot_kfold_'+kfold+'_my_net.25sup.pkl'
save_file = '/1_shot_snips_.woquery.pkl.novel.'+str(raw_gen_support)+'sup'
print()

def plot_with_labels(low_dim_embs, labels,i):
    x_min, x_max = np.min(low_dim_embs, 0), np.max(low_dim_embs, 0)
    low_dim_embs = (low_dim_embs-x_min) / (x_max - x_min)
    plt.cla()
    # 降到二维了，分别给x和y
    X, Y = low_dim_embs[:, 0], low_dim_embs[:, 1]
    # fig = plt.figure()

    # plt.scatter(X,Y,c=labels,s=1,cmap=plt.cm.get_cmap('jet',10))
    # plt.colorbar(ticks=range(10))
    # plt.clim(-1,1)
    # plt.show()
    # 遍历每个点以及对应标签
    plt.figure(figsize=(6, 5))
    length = 30
    plt.scatter(X[length*0:length*1], Y[length*0:length*1], s=10, c='darkgrey', marker='o')
    plt.scatter(X[length*1:length*2], Y[length*1:length*2], s=10, c='darkgreen', marker='o')
    plt.scatter(X[length*2:length*3], Y[length*2:length*3], s=10, c='darkblue', marker='o')
    plt.scatter(X[length*3:length*4], Y[length*3:length*4], s=10, c='darkgoldenrod', marker='o')
    plt.scatter(X[length*4:length*5], Y[length*4:length*5], s=10, c='darkred', marker='o')
    plt.scatter(X[length*5:length*6], Y[length*5:length*6], s=10, c='darkviolet', marker='o')
    plt.scatter(X[length*6:length*7], Y[length*6:length*7], s=10, c='deepskyblue', marker='o')


    # for x, y, s in zip(X, Y, labels):
    #     c = cm.rainbow(int(255/9 * s)) # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
    #     # plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    #     plt.scatter(x,y,c=c,s=20)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    # plt.title('Visualize last layer')
    plt.legend(['play music', 'search creative work', 'search screening event','book restaurant','get weather','add to playlist','rate book'])
    plt.savefig("fig/5_shot_snips_rawnet.svg")
    plt.show()

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


class ProtoNet(nn.Module):
    def __init__(self, encoder,generator, metric="euclidean"):
        super(ProtoNet, self).__init__()

        self.encoder = encoder
        self.generator = generator
        self.metric = metric
        self.LayerNorm = CNTALayerNorm(768)
        assert self.metric in ('euclidean', 'cosine')

    def loss(self, sample,target_inds_input=None):
        """
        :param sample: {
            "xs": [
                [support_A_1, support_A_2, ...],
                [support_B_1, support_B_2, ...],
                [support_C_1, support_C_2, ...],
                ...
            ],
            "xq": [
                [query_A_1, query_A_2, ...],
                [query_B_1, query_B_2, ...],
                [query_C_1, query_C_2, ...],
                ...
            ],
            "xg": [
                [gen_support_A_1, gen_support_A_2],
                [gen_support_B_1, gen_support_B_2],
                [gen_support_C_1, gen_support_C_2],
                ...
                成对出现
            ]
        }
        :return:
        """
        loss_func = nn.MSELoss()
        xs = sample['xs']  # support
        xq = sample['xq']  # query
        xg = sample['xg']  # generator

        n_gen_support = len(xg)
        n_class = len(xs)
        n_shot = len(xs[0])
        # assert len(xq) == n_class
        n_support = len(xs[0])
        n_query = len(xq[0])

        if target_inds_input == None:
            target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
            target_inds = Variable(target_inds, requires_grad=False).to(device)
        else:
            target_inds = target_inds_input

        x = [item for xs_ in xs for item in xs_] + [item for xg_ in xg for item in xg_] + [item for xq_ in xq for item in xq_]
        _ , z,sequence_output = self.encoder.forward(x)
        # z = self.LayerNorm(z)
        z_dim = z.size(-1)

        loss_1 = 0
        # 此处添加generator代码,去除ff
        # z_support = z[:n_class * n_support]
        # zq = z[n_class * n_support + n_gen_support * 2:]
        # zg = z[n_class * n_support : n_class * n_support + n_gen_support * 2]
        # zg = zg.view(n_gen_support,2,768)
        # zg_1 = zg[:,0,:]
        # zg_2 = zg[:,1,:]
        # weight = torch.zeros((n_class, 768), requires_grad=True).to(device)
        # for i in range(n_class):
        #     weight_point = torch.zeros(n_shot * (n_gen_support + 1), 768)
        #     for j in range(n_shot):
        #         a = z_support[i * n_shot + j]
        #         gen_feature = self.generator(zg_1, zg_2, a)
        #         features = torch.cat((gen_feature,a.unsqueeze(0)),0)
        #         weight_point[j * (n_gen_support + 1):(j + 1) * (n_gen_support + 1)] = features
        #     weight[i] = torch.mean(weight_point, 0)
        # z_proto = weight
        #
        # if target_inds_input==None:
        #     weight_ = torch.zeros((n_class * n_query, 768), requires_grad=True).to(device)
        #     for i in range(n_class):
        #         for j in range(n_query):
        #             a = zq[i * n_query + j]
        #             gen_feature = self.generator(zg_1, zg_2, a)
        #             features = torch.cat((gen_feature, a.unsqueeze(0)), 0)
        #             features = torch.mean(features,0)
        #             weight_[i * n_query + j] = features
        # else:
        #     weight_ = torch.zeros((n_query, 768), requires_grad=True).to(device)
        #     for j in range(n_query):
        #         a = zq[j]
        #         gen_feature = self.generator(zg_1, zg_2, a)
        #         features = torch.cat((gen_feature, a.unsqueeze(0)), 0)
        #         features = torch.mean(features, 0)
        #         weight_[j] = features
        # zq = weight_



        # 增强support和query
        z_support = sequence_output[:n_class * n_support]
        zq = sequence_output[n_class * n_support + n_gen_support * 2:]
        zg = sequence_output[n_class * n_support: n_class * n_support + n_gen_support * 2]
        zg = zg.view(n_gen_support, 2, 30,768)
        zg_1 = zg[:, 0, :,:]
        zg_2 = zg[:, 1, :,:]
        weight = torch.zeros((n_class, 768*2), requires_grad=True).to(device)
        for i in range(n_class):
            weight_point = torch.zeros(n_shot*(n_gen_support+1), 768*2)
            for j in range(n_shot):
                a = z_support[i*n_shot+j]
                gen_feature = self.generator(zg_1, zg_2, a)
                gen_feature = torch.mean(gen_feature,0)
                gen_feature = torch.cat((gen_feature.unsqueeze(0), a.unsqueeze(0)), 0)
                features = self.encoder.bert.pooler(gen_feature)
                features = torch.cat((features[0].unsqueeze(0), features[1].unsqueeze(0)), 1)
                weight_point[j*(n_gen_support+1):(j+1)*(n_gen_support+1)] = features
            weight[i] = torch.mean(weight_point, 0)
        z_proto = weight


        # 此处为对query也做变换
        if target_inds_input==None:
            weight_ = torch.zeros((n_class * n_query, 768*2), requires_grad=True).to(device)
            for i in range(n_class):
                for j in range(n_query):
                    a = zq[i * n_query + j]
                    gen_feature = self.generator(zg_1, zg_2, a)
                    gen_feature = torch.mean(gen_feature, 0)

                    # gen_feature = torch.cat((gen_feature.unsqueeze(0), a.unsqueeze(0)), 0)
                    # 去掉gen_feature
                    gen_feature = torch.cat((a.unsqueeze(0), a.unsqueeze(0)), 0)

                    features = self.encoder.bert.pooler(gen_feature)
                    features = torch.cat((features[0].unsqueeze(0), features[1].unsqueeze(0)), 1)
                    weight_[i * n_query + j] = features
        else:
            weight_ = torch.zeros((n_query, 768*2), requires_grad=True).to(device)
            for j in range(n_query):
                a = zq[j]
                gen_feature = self.generator(zg_1, zg_2, a)
                gen_feature = torch.mean(gen_feature, 0)

                # gen_feature = torch.cat((gen_feature.unsqueeze(0), a.unsqueeze(0)), 0)
                # 去掉gen_feature
                gen_feature = torch.cat((a.unsqueeze(0), a.unsqueeze(0)), 0)

                features = self.encoder.bert.pooler(gen_feature)
                features = torch.cat((features[0].unsqueeze(0), features[1].unsqueeze(0)), 1)
                weight_[j] = features
        zq = weight_

        # z_proto相当于将样本平均，得到类向量
        # z_proto = z[:n_class * n_support].view(n_class, n_support, z_dim).mean(1)
        # zq = z[n_class * n_support + n_gen_support * 2:]

        # # 绘制向量降维图
        # tsne = TSNE(n_components=2, n_jobs=4)  # TSNE降维，降到2
        # low_dim_embs = tsne.fit_transform(zq.cpu().detach().numpy())
        # labels = np.array(target_inds)
        # plot_with_labels(low_dim_embs, labels, 1)

        if self.metric == "euclidean":
            dists = euclidean_dist(zq, z_proto)
        elif self.metric == "cosine":
            dists = (-cosine_similarity(zq, z_proto) + 1) * 5
        else:
            raise NotImplementedError

        if target_inds_input == None:
            log_p_y = torch_functional.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
            dists.view(n_class, n_query, -1)
            loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean() + loss_1
            # 此处与上述作用等价，loss计算结果相同
            # criterion = nn.CrossEntropyLoss()
            # loss_val = criterion(log_p_y.view(-1, n_class), target_inds.view(-1))
            _, y_hat = log_p_y.max(2)
            acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
            return loss_val, {
                'loss': loss_val.item(),
                'acc': acc_val.item(),
                'dists': dists,
                'target': target_inds
            }
        else:
            log_p_y = torch_functional.log_softmax(-dists, dim=1)
            _,y_hat = log_p_y.max(1)
            correct_num = torch.eq(y_hat, target_inds.view(-1)).sum().float().item()
            return correct_num

    def loss_softkmeans(self, sample):
        xs = sample['xs']  # support
        xq = sample['xq']  # query
        xu = sample['xu']  # unlabeled

        n_class = len(xs)
        assert len(xq) == n_class
        n_support = len(xs[0])
        n_query = len(xq[0])

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False).to(device)

        x = [item for xs_ in xs for item in xs_] + [item for xq_ in xq for item in xq_] + [item for item in xu]
        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        zs = z[:n_class * n_support]
        z_proto = z[:n_class * n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class * n_support: (n_class * n_support) + (n_class * n_query)]
        zu = z[(n_class * n_support) + (n_class * n_query):]

        distances_to_proto = euclidean_dist(
            torch.cat((zs, zu)),
            z_proto
        )

        distances_to_proto_normed = torch.nn.Softmax(dim=-1)(-distances_to_proto)

        refined_protos = list()
        for class_ix in range(n_class):
            z = torch.cat(
                (zs[class_ix * n_support: (class_ix + 1) * n_support], zu)
            )
            d = torch.cat(
                (torch.ones(n_support).to(device),
                 distances_to_proto_normed[(n_class * n_support):, class_ix])
            )
            refined_proto = ((z.t() * d).sum(1) / d.sum())
            refined_protos.append(refined_proto.view(1, -1))
        refined_protos = torch.cat(refined_protos)

        if self.metric == "euclidean":
            dists = euclidean_dist(zq, refined_protos)
        elif self.metric == "cosine":
            dists = (-cosine_similarity(zq, refined_protos) + 1) * 5
        else:
            raise NotImplementedError

        log_p_y = torch_functional.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
        dists.view(n_class, n_query, -1)
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'dists': dists,
            'target': target_inds
        }

    def train_step(self, optimizer, data_dict: Dict[str, List[str]],
                   sup_data_dict: Dict[str, List[str]],
                   train_support_data_dict,
                   n_support: int, n_classes: int, n_query: int, n_unlabeled: int,
                   ):

        # episode = create_episode(
        #     data_dict=data_dict,
        #     sup_data_dict = sup_data_dict,
        #     n_support=n_support,
        #     n_classes=n_classes,
        #     n_query=n_query,
        #     n_gen_support=40,
        #     n_unlabeled=n_unlabeled
        # )

        episode = create_snips_episode(
            target_query_dict=data_dict,
            target_support_dict=train_support_data_dict,
            sup_data_dict=sup_data_dict,
            n_support=n_support,
            n_classes=n_classes,
            n_query=n_query,
            n_gen_support=raw_gen_support,
            n_unlabeled=n_unlabeled
        )

        self.train()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        if n_unlabeled:
            loss, loss_dict = self.loss_softkmeans(episode)
        else:
            loss, loss_dict = self.loss(episode)
        loss.backward()
        optimizer.step()

        return loss, loss_dict

    def test_step(self, data_dict,
                  sup_data_dict,
                  target_support_data_dict,
                  n_support, n_classes, n_query, n_unlabeled=0, n_episodes=1000):
        accuracies = list()
        losses = list()
        self.eval()
        # target_query_dict = deepcopy(data_dict)
        for i in range(int(n_episodes)):
            episode = create_snips_episode(
                target_query_dict=data_dict,
                target_support_dict = target_support_data_dict,
                sup_data_dict=sup_data_dict,
                n_support=n_support,
                n_classes=n_classes,
                n_query=n_query,
                n_gen_support=raw_gen_support,
                n_unlabeled=n_unlabeled
            )
            with torch.no_grad():
                if n_unlabeled:
                    loss, loss_dict = self.loss_softkmeans(episode)
                else:
                    loss, loss_dict = self.loss(episode)

            accuracies.append(loss_dict["acc"])
            losses.append(loss_dict["loss"])

        return {
            "loss": np.mean(losses),
            "acc": np.mean(accuracies)
        }

def run_proto(
        train_path: str,
        model_name_or_path: str,
        n_support: int,
        n_query: int,
        n_classes: int,
        valid_path: str = None,
        test_path: str = None,
        sup_path: str = None,
        target_support_path: str=None,
        train_support_path: str=None,
        n_unlabeled: int = 0,
        output_path: str = f'runs/{now()}',
        max_iter: int = 10000,
        evaluate_every: int = 100,
        early_stop: int = None,
        n_test_episodes: int = 1000,
        log_every: int = 10,
        metric: str = "euclidean",
        arsc_format: bool = False,
        data_path:str=None,
        tgt:str='joint'
):
    # if output_path:
    #     if os.path.exists(output_path) and len(os.listdir(output_path)):
    #         raise FileExistsError(f"Output path {output_path} already exists. Exiting.")

    # --------------------
    # Creating Log Writers
    # --------------------
    # os.makedirs(output_path)
    # os.makedirs(os.path.join(output_path, "logs/train"))
    train_writer: SummaryWriter = SummaryWriter(logdir=os.path.join(output_path, "logs/train"), flush_secs=1, max_queue=1)
    valid_writer: SummaryWriter = None
    test_writer: SummaryWriter = None
    log_dict = dict(train=list())

    if valid_path:
        # os.makedirs(os.path.join(output_path, "logs/valid"))
        valid_writer = SummaryWriter(logdir=os.path.join(output_path, "logs/valid"), flush_secs=1, max_queue=1)
        log_dict["valid"] = list()
    if test_path:
        # os.makedirs(os.path.join(output_path, "logs/test"))
        test_writer = SummaryWriter(logdir=os.path.join(output_path, "logs/test"), flush_secs=1, max_queue=1)
        log_dict["test"] = list()

    def raw_data_to_labels_dict(data, shuffle=True):
        labels_dict = collections.defaultdict(list)
        for item in data:
            labels_dict[item['label']].append(item["sentence"])
        labels_dict = dict(labels_dict)
        if shuffle:
            for key, val in labels_dict.items():
                random.shuffle(val)
        return labels_dict

    # Load model
    bert_classifier = BertClassifier(model_name_or_path, num_labels=n_classes).to(device)
    # bert_classifier.load_state_dict(torch.load('../DTN/save/mynet.nlue_1.1_shot.classifier.finetune.pkl'))
    generator = GeneratorNet().to(device)
    # generator.load_state_dict(torch.load('../DTN/save/mynet.nlue_1.1_shot.generator.finetune.pkl'))

    # bert = BERTEncoder(model_name_or_path).to(device)
    # generator = GeneratorNet().to(device)
    protonet = ProtoNet(encoder=bert_classifier, generator=generator,metric=metric).to(device)
    # protonet.load_state_dict(torch.load('save/1_shot_kfold_'+kfold+'_my_net.pkl.10'))
    # protonet.load_state_dict(torch.load('runs/2021-04-13_12:18:39.266432/protonet.pkl'))

    # optimizer = torch.optim.Adam(protonet.parameters(), lr=2e-5)
    optimizer = torch.optim.Adam([
        {'params': protonet.encoder.parameters(), 'lr': 2e-5},
        {'params': protonet.generator.parameters(), 'lr': 2e-5},
    ]
    )

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.65)

    # Load data
    if not arsc_format:
        train_data = get_jsonl_data(train_path)
        train_data_dict = raw_data_to_labels_dict(train_data, shuffle=True)
        logger.info(f"train labels: {train_data_dict.keys()}")

        if valid_path:
            valid_data = get_jsonl_data(valid_path)
            valid_data_dict = raw_data_to_labels_dict(valid_data, shuffle=True)
            logger.info(f"valid labels: {valid_data_dict.keys()}")
        else:
            valid_data_dict = None

        if test_path:
            test_data = get_jsonl_data(test_path)
            test_data_dict = raw_data_to_labels_dict(test_data, shuffle=True)
            logger.info(f"test labels: {test_data_dict.keys()}")
        else:
            test_data_dict = None

        if sup_path:
            sup_data = get_jsonl_data(sup_path)
            sup_data_dict = raw_data_to_labels_dict(sup_data, shuffle=True)
            # 只保留固定的20对样本
            # temp = {}
            # for key in sup_data_dict:
            #     temp[key] = sup_data_dict[key][:2]
            # sup_data_dict = temp
            logger.info(f"sup labels: {sup_data_dict.keys()}")
        else:
            sup_data_dict = None

        if target_support_path:
            target_support_data = get_jsonl_data(target_support_path)
            target_support_data_dict = raw_data_to_labels_dict(target_support_data, shuffle=True)
            logger.info(f"target labels: {target_support_data_dict.keys()}")
        else:
            target_support_data_dict = None

        if train_support_path:
            train_support_data = get_jsonl_data(train_support_path)
            train_support_data_dict = raw_data_to_labels_dict(train_support_data, shuffle=True)
            logger.info(f"train support labels: {train_support_data_dict.keys()}")
        else:
            train_support_data_dict = None
    else:
        train_data_dict = None
        valid_data_dict = None
        test_data_dict = None
        sup_data_dict = None

    train_accuracies = list()
    train_losses = list()
    n_eval_since_last_best = 0
    best_valid_acc = 0.0
    # np.random.seed(raw_gen_support)
    if tgt == 'joint':
        n_classes = 2 * n_classes
    for step in tqdm.tqdm(range(max_iter)):
        # scheduler.step()
        loss, loss_dict = protonet.train_step(
            optimizer=optimizer,
            data_dict=train_data_dict,
            sup_data_dict = sup_data_dict,
            train_support_data_dict = train_support_data_dict,
            n_unlabeled=n_unlabeled,
            n_support=n_support,
            n_query=n_query,
            n_classes=n_classes
        )
        train_accuracies.append(loss_dict["acc"])
        train_losses.append(loss_dict["loss"])

        # Logging
        if (step + 1) % log_every == 0:
            train_writer.add_scalar(tag="loss", scalar_value=np.mean(train_losses), global_step=step)
            train_writer.add_scalar(tag="accuracy", scalar_value=np.mean(train_accuracies), global_step=step)
            logger.info(f"train | loss: {np.mean(train_losses):.4f} | acc: {np.mean(train_accuracies):.4f}")
            log_dict["train"].append({
                "metrics": [
                    {
                        "tag": "accuracy",
                        "value": np.mean(train_accuracies)
                    },
                    {
                        "tag": "loss",
                        "value": np.mean(train_losses)
                    }

                ],
                "global_step": step
            })

            train_accuracies = list()
            train_losses = list()

        if valid_path or test_path:
            if (step + 1) % evaluate_every == 0:
                torch.save(protonet.state_dict(),output_path+'/protonet.pkl')
                for path, writer, set_type, set_data in zip(
                        [valid_path, test_path],
                        [valid_writer, test_writer],
                        ["valid", "test"],
                        [valid_data_dict, test_data_dict]
                ):
                    if path:
                        if not arsc_format:
                            set_results = protonet.test_step(
                                data_dict=set_data,
                                sup_data_dict=sup_data_dict,
                                target_support_data_dict=target_support_data_dict,
                                n_unlabeled=n_unlabeled,
                                n_support=n_support,
                                n_query=n_query,
                                n_classes=n_classes,
                                n_episodes=n_test_episodes
                            )
                        else:
                            set_results = protonet.test_step_ARSC(
                                data_path=data_path,
                                n_unlabeled=n_unlabeled,
                                n_episodes=n_test_episodes,
                                set_type={"valid": "dev", "test": "test"}[set_type]
                            )

                        writer.add_scalar(tag="loss", scalar_value=set_results["loss"], global_step=step)
                        writer.add_scalar(tag="accuracy", scalar_value=set_results["acc"], global_step=step)
                        log_dict[set_type].append({
                            "metrics": [
                                {
                                    "tag": "accuracy",
                                    "value": set_results["acc"]
                                },
                                {
                                    "tag": "loss",
                                    "value": set_results["loss"]
                                }

                            ],
                            "global_step": step
                        })

                        logger.info(f"{set_type} | loss: {set_results['loss']:.4f} | acc: {set_results['acc']:.4f}")
                        if set_type == "valid":
                            if set_results["acc"] > best_valid_acc:
                                best_valid_acc = set_results["acc"]
                                n_eval_since_last_best = 0
                                logger.info(f"Better eval results!")
                            else:
                                n_eval_since_last_best += 1
                                logger.info(f"Worse eval results ({n_eval_since_last_best}/{early_stop})")

                if early_stop and n_eval_since_last_best >= early_stop:
                    logger.warning(f"Early-stopping.")
                    break

    torch.save(protonet.state_dict(), output_path + save_file)

    # with open(os.path.join(output_path, 'metrics.json'), "w") as file:
    #     json.dump(log_dict, file, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--train-path", type=str, default='../../dataset/trans_dataset/NLUE/KFold_'+kfold+'/base_samples.jsonl', help="Path to training data")
    parser.add_argument("--train-path", type=str,
                        default='../../dataset/trans_dataset/SNIPS/base_samples.jsonl',
                        help="Path to base samples")

    parser.add_argument("--valid-path", type=str, default=None, help="Path to validation data")
    # parser.add_argument("--test-path", type=str, default='../../dataset/trans_dataset/NLUE/KFold_'+kfold+'/test_novel_data.jsonl', help="Path to testing data")
    parser.add_argument("--test-path", type=str,default='../../dataset/trans_dataset/SNIPS/test_novel_data.jsonl',help="Path to testing data")
    # parser.add_argument("--train-support-path", type=str,
    #                     default='../../dataset/trans_dataset/NLUE/KFold_'+kfold+'/base_samples.jsonl')
    parser.add_argument("--train-support-path", type=str,
                        default='../../dataset/trans_dataset/SNIPS/base_samples.jsonl')
    # parser.add_argument("--target-support-path", type=str, default='../../dataset/trans_dataset/NLUE/KFold_'+kfold+'/support_1_shots_novel_data.jsonl')
    parser.add_argument("--target-support-path", type=str,default='../../dataset/trans_dataset/SNIPS/support_1_shots_novel_data.jsonl')

    parser.add_argument("--data-path", type=str, default=None, help="Path to data (ARSC only)")
    # parser.add_argument('--sup-path',type=str, default='../../dataset/trans_dataset/NLUE/KFold_'+kfold+'/auxiliary_samples.jsonl', help="Path to sup data")
    parser.add_argument('--sup-path', type=str,
                        default='../../dataset/trans_dataset/SNIPS/auxiliary_samples.jsonl',
                        help="Path to auxiliary samples")

    parser.add_argument("--output-path", type=str, default='save')
    # parser.add_argument("--model-name-or-path", type=str,
    #                     default='/opt/yfy/PythonProjects/DFEPN/language_modeling/transformer_models/nlue/kfold_'+kfold+'/fine-tuned/',
    #                     help="Transformer model to use")
    parser.add_argument("--model-name-or-path", type=str,
                        default='../../language_modeling/transformer_models/snips/fine-tuned/',
                        help="Transformer model to use")
    # parser.add_argument("--model-name-or-path", type=str,default='/opt/yfy/PythonProjects/Fewshot-SLU/data/bert/bert-base-uncased/', help="Transformer model to use")

    parser.add_argument("--n-unlabeled", type=int, help="Number of unlabeled data points per class (proto++)", default=0)
    parser.add_argument("--max-iter", type=int, default=1000, help="Max number of training episodes")
    parser.add_argument("--evaluate-every", type=int, default=10000, help="Number of training episodes between each evaluation (on both valid, test)")
    parser.add_argument("--log-every", type=int, default=10, help="Number of training episodes between each logging")
    parser.add_argument("--seed", type=int, default=5, help="Random seed to set")
    parser.add_argument("--early-stop", type=int, default=0, help="Number of worse evaluation steps before stopping. 0=disabled")

    # Few-Shot related stuff
    parser.add_argument("--n-support", type=int, default=1, help="Number of support points for each class")
    parser.add_argument("--n-query", type=int, default=20, help="Number of query points for each class")
    parser.add_argument("--n-classes", type=int, default=2, help="Number of classes per episode")
    parser.add_argument("--n-test-episodes", type=int, default=1000, help="Number of episodes during evaluation (valid, test)")

    # Metric to use in proto distance calculation
    parser.add_argument("--metric", type=str, default="euclidean", help="Metric to use", choices=("euclidean", "cosine"))


    parser.add_argument("--arsc-format", default=False, action="store_true", help="Using ARSC few-shot format")
    parser.add_argument("--eps-format", default=True, action="store_true", help="eps or noneps")
    parser.add_argument("--predict", default=True, action="store_true", help="if predict")
    parser.add_argument("--tgt",type=str, default='novel',choices=('novel','joint'))

    args = parser.parse_args()

    # Set random seed
    set_seeds(args.seed)

    # Check if data path(s) exist
    for arg in [args.train_path, args.valid_path, args.test_path]:
        if arg and not os.path.exists(arg):
            raise FileNotFoundError(f"Data @ {arg} not found.")

    # Run
    run_proto(
        train_path=args.train_path,
        valid_path=args.valid_path,
        test_path=args.test_path,
        output_path=args.output_path,
        sup_path = args.sup_path,
        train_support_path = args.train_support_path,
        target_support_path = args.target_support_path,

        model_name_or_path=args.model_name_or_path,
        n_unlabeled=args.n_unlabeled,

        n_support=args.n_support,
        n_query=args.n_query,
        n_classes=args.n_classes,
        n_test_episodes=args.n_test_episodes,

        max_iter=args.max_iter,
        evaluate_every=args.evaluate_every,

        metric=args.metric,
        early_stop=args.early_stop,
        arsc_format=args.arsc_format,
        data_path=args.data_path,
        log_every=args.log_every,
        tgt = args.tgt
    )

    # Save config
    with open(os.path.join(args.output_path, "config.json"), "w") as file:
        json.dump(vars(args), file, ensure_ascii=False)


if __name__ == '__main__':
    main()
