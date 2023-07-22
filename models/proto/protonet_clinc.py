import sys
sys.path.append('/opt/yfy/PythonProjects/DFEPN')

import json
import argparse
from models.encoders.bert_encoder import BERTEncoder
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
from utils.few_shot import create_episode,create_snips_episode,create_clinc_episode
from utils.my_math import euclidean_dist, cosine_similarity
from models.DTN.generator import GeneratorNet
import tqdm
from copy import deepcopy
from models.DTN.bert_classifier import BertClassifier

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.simplefilter('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ProtoNet(nn.Module):
    def __init__(self, encoder,generator, generator_query,metric="euclidean"):
        super(ProtoNet, self).__init__()

        self.encoder = encoder
        self.generator = generator
        self.generator_query = generator_query
        self.metric = metric
        assert self.metric in ('euclidean', 'cosine')

    def loss(self, sample,criterion):
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
        xs = sample['xs']  # support
        xq = sample['xq']  # query
        xg = sample['xg']  # generator

        n_gen_support = len(xg)
        n_class = len(xs)
        n_shot = len(xs[0])
        assert len(xq) == n_class
        n_support = len(xs[0])
        n_query = len(xq[0])

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False).to(device)

        x = [item for xs_ in xs for item in xs_] + [item for xq_ in xq for item in xq_] + [item for xg_ in xg for item in xg_]
        _,z = self.encoder.forward(x)
        z_dim = z.size(-1)


        # z_support = z[:n_class * n_support]
        # zq = z[n_class * n_support: n_class * n_support + n_class * n_query]
        # zg = z[n_class * n_support + n_class * n_query:]
        # zg = zg.view(n_gen_support, 2, 768)
        # zg_1 = zg[:, 0, :]
        # zg_2 = zg[:, 1, :]
        # weight = torch.zeros((n_class, 768), requires_grad=True).to(device)
        #
        # zg_1 = self.encoder.bert.pooler(zg_1.unsqueeze(1))
        # zg_2 = self.encoder.bert.pooler(zg_2.unsqueeze(1))
        # for i in range(n_class):
        #     weight_point = torch.zeros(n_shot * (n_gen_support + 1), 768)
        #     for j in range(n_shot):
        #         a = z_support[i * n_shot + j]
        #         a = self.encoder.bert.pooler(a.unsqueeze(1))
        #         gen_feature = zg_1 - zg_2 + a
        #         gen_feature = self.encoder.bert.pooler(gen_feature.unsqueeze(1))
        #
        #         features = torch.cat((gen_feature + a, a.unsqueeze(0)), 0)
        #         weight_point[j * (n_gen_support + 1):(j + 1) * (n_gen_support + 1)] = features
        #     weight[i] = torch.mean(weight_point, 0)
        # z_proto = weight
        #
        # weight_ = torch.zeros((n_class * n_query, 768), requires_grad=True).to(device)
        # for i in range(n_class):
        #     weight_point_ = torch.zeros(1 * (n_gen_support + 1), 768)
        #     for j in range(n_query):
        #         a = zq[i * n_query + j]
        #         a = self.encoder.bert.pooler(a.unsqueeze(1))
        #         # gen_feature = zg_1 - zg_2
        #         # gen_feature = self.encoder.bert.pooler(gen_feature.unsqueeze(1))
        #         # features = torch.cat((gen_feature + a, a.unsqueeze(0)), 0)
        #         # weight_point_ = features
        #         # weight_[i * n_query + j] = torch.mean(weight_point_, 0)
        #         weight_[i * n_query + j] = a
        # zq = weight_


        # 此处添加generator代码
        z_support = z[:n_class * n_support]
        zq = z[n_class * n_support : n_class * n_support + n_class * n_query]
        zg = z[n_class * n_support + n_class * n_query :]
        zg = zg.view(n_gen_support,2,768)
        zg_1 = zg[:,0,:]
        zg_2 = zg[:,1,:]
        # W_base = self.encoder.classifier.weight.data
        # probs = torch.matmul(zq, torch.transpose(W_base, 0, 1))
        # predict_1 = torch.max(probs, 1)[1]
        # probs = torch.matmul(z_support, torch.transpose(W_base, 0, 1))
        # predict_2 = torch.max(probs, 1)[1]
        # zg_query = torch.zeros((n_class,2,768),requires_grad=True).to(device)
        weight = torch.zeros((n_class, 768), requires_grad=True).to(device)
        for i in range(n_class):
            weight_point = torch.zeros(n_shot*(n_gen_support+1), 768)
            for j in range(n_shot):
                a = z_support[i*n_shot+j]
                gen_feature = self.generator(zg_1, zg_2, a)
                features = torch.cat((gen_feature, a.unsqueeze(0)))
                # features = torch.cat((gen_feature, a.unsqueeze(0)))
                weight_point[j*(n_gen_support+1):(j+1)*(n_gen_support+1)] = features
            # rand_q = random.sample(range(0, n_shot*(n_gen_support+1)), 2)
            # zg_query[i][0] = weight_point[rand_q[0]]
            # zg_query[i][1] = weight_point[rand_q[1]]
            weight[i] = torch.mean(weight_point, 0)
        z_proto = weight

        # 此处为对query也做变换
        # zg_1_query = zg_query[:, 0, :]
        # zg_2_query = zg_query[:, 1, :]
        # weight_ = torch.zeros((n_class * n_query, 768), requires_grad=True).to(device)
        # for i in range(n_class):
        #     for j in range(n_query):
        #         a = zq[i * n_query + j]
        #         gen_feature = self.generator_query(zg_1, zg_2, a)
        #         features = torch.cat((gen_feature, a.unsqueeze(0)))
        #         weight_point_ = features
        #         weight_[i * n_query + j] = torch.mean(weight_point_, 0)
        # zq = weight_



        # z_proto相当于将样本平均，得到类向量
        # z_proto = z[:n_class * n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class * n_support : n_class * n_support + n_class * n_query]


        if self.metric == "euclidean":
            dists = euclidean_dist(zq, z_proto)
            # dists = self.l2_norm(dists)
        elif self.metric == "cosine":
            dists = (-cosine_similarity(zq, z_proto) + 1) * 5
        else:
            raise NotImplementedError

        log_p_y = torch_functional.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
        dists.view(n_class, n_query, -1)
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()


        # 此处与上述作用等价，loss计算结果相同
        # probs = torch.matmul(zq, torch.transpose(z_proto, 0, 1))
        # predict = torch.max(probs, 1)[1]
        # loss_val = criterion(probs.view(-1, n_class), target_inds.view(-1))
        # correct = torch.eq(predict, target_inds.view(-1)).sum().float().item()
        # acc_val = correct / len(target_inds.view(-1))

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            # 'acc': acc_val,
            'dists': dists,
            'target': target_inds
        }
    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        norm = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(norm)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

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
                   criterion,
                   n_support: int, n_classes: int, n_query: int, n_unlabeled: int):

        episode = create_episode(
            data_dict=data_dict,
            sup_data_dict = sup_data_dict,
            n_support=n_support,
            n_classes=n_classes,
            n_query=n_query,
            n_gen_support=40,
            n_unlabeled=n_unlabeled
        )

        # episode = create_snips_episode(
        #     target_query_dict=data_dict,
        #     target_support_dict=train_support_data_dict,
        #     sup_data_dict=sup_data_dict,
        #     n_support=n_support,
        #     n_classes=n_classes,
        #     n_query=n_query,
        #     n_gen_support=20,
        #     n_unlabeled=n_unlabeled
        # )
        self.train()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        if n_unlabeled:
            loss, loss_dict = self.loss_softkmeans(episode,criterion)
        else:
            loss, loss_dict = self.loss(episode,criterion)
        loss.backward()
        optimizer.step()
        return loss, loss_dict

    def test_step(self, data_dict,
                  sup_data_dict,
                  target_support_data_dict,
                  criterion,
                  n_support, n_classes, n_query, n_unlabeled=0, n_episodes=1000):
        accuracies = list()
        losses = list()
        self.eval()
        # target_query_dict = deepcopy(data_dict)
        for i in range(int(n_episodes)):
            episode_list = create_clinc_episode(
                target_query_dict=data_dict,
                target_support_dict=target_support_data_dict,
                sup_data_dict=sup_data_dict,
                n_support=n_support,
                n_classes=n_classes,
                n_query=n_query,
                n_gen_support=40,
                n_unlabeled=n_unlabeled
            )
            with torch.no_grad():
                if n_unlabeled:
                    loss, loss_dict = self.loss_softkmeans(episode_list)
                else:
                    for episode in episode_list:
                        loss, loss_dict = self.loss(episode,criterion)
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
        data_path:str=None
):
    if output_path:
        if os.path.exists(output_path) and len(os.listdir(output_path)):
            raise FileExistsError(f"Output path {output_path} already exists. Exiting.")

    # --------------------
    # Creating Log Writers
    # --------------------
    os.makedirs(output_path)
    os.makedirs(os.path.join(output_path, "logs/train"))
    train_writer: SummaryWriter = SummaryWriter(logdir=os.path.join(output_path, "logs/train"), flush_secs=1, max_queue=1)
    valid_writer: SummaryWriter = None
    test_writer: SummaryWriter = None
    log_dict = dict(train=list())

    if valid_path:
        os.makedirs(os.path.join(output_path, "logs/valid"))
        valid_writer = SummaryWriter(logdir=os.path.join(output_path, "logs/valid"), flush_secs=1, max_queue=1)
        log_dict["valid"] = list()
    if test_path:
        os.makedirs(os.path.join(output_path, "logs/test"))
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
    # bert_path = '/opt/yfy/PythonProjects/Fewshot-SLU/data/bert/bert-base-uncased/'
    bert_path = '/opt/yfy/PythonProjects/DFEPN/language_modeling/transformer_models/clinc150/banking/fine-tuned/'
    bert = BertClassifier(bert_path,num_labels=150).to(device)
    # bert.load_state_dict(torch.load(model_name_or_path))

    generator = GeneratorNet().to(device)
    # generator.load_state_dict(torch.load('../DTN/save/bert_classifier_40.pkl'))
    generator_query = GeneratorNet().to(device)
    # generator_query.load_state_dict(torch.load('../DTN/save/bert_classifier_40.pkl'))
    protonet = ProtoNet(encoder=bert, generator=generator,generator_query=generator_query,metric=metric)
    # protonet.load_state_dict(torch.load('clinc_5_5_cos/protonet.pkl'))

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(protonet.parameters(), lr=2e-5)
    optimizer = torch.optim.Adam(
        [
            {'params': protonet.encoder.parameters(), 'lr': 2e-5},
            {'params': protonet.generator.parameters(), 'lr': 1e-4},
            {'params': protonet.generator_query.parameters(), 'lr': 1e-4},
        ],
        lr=1e-4,
    )

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

    for step in tqdm.tqdm(range(max_iter)):
        if not arsc_format:
            loss, loss_dict = protonet.train_step(
                optimizer=optimizer,
                data_dict=train_data_dict,
                sup_data_dict = sup_data_dict,
                train_support_data_dict = train_support_data_dict,
                criterion=criterion,
                n_unlabeled=n_unlabeled,
                n_support=n_support,
                n_query=n_query,
                n_classes=n_classes
            )
        else:
            loss, loss_dict = protonet.train_step_ARSC(
                optimizer=optimizer,
                n_unlabeled=n_unlabeled,
                data_path=data_path
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
                                criterion=criterion,
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
    with open(os.path.join(output_path, 'metrics.json'), "w") as file:
        json.dump(log_dict, file, ensure_ascii=False)

'''
C way K shot Q query
1、因为微调时是对train.jsonl的句子进行的，因此将sup-path也设为train里的句子进行选择
2、测试时从target.train.jsonl中选C*K个句子作为support set
3、从target.test.jsonl中将所有属于C类的样本挑出，Q个一组进行测试
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, default='../../dataset/trans_dataset/clinc150/banking/meta_data.jsonl', help="Path to training data")
    # parser.add_argument("--valid-path", type=str, default='../../dataset/trans_dataset/clinc150/target.dev.jsonl', help="Path to validation data")
    parser.add_argument("--valid-path", type=str, default=None, help="Path to validation data")
    parser.add_argument("--test-path", type=str, default='../../dataset/trans_dataset/clinc150/banking/target.test.jsonl', help="Path to testing data")
    parser.add_argument("--train-support-path", type=str,
                        default=None)
    # parser.add_argument("--target-support-path", type=str, default='../../dataset/trans_dataset/clinc150/banking/target.train.jsonl')
    parser.add_argument("--target-support-path", type=str,
                        default='../../dataset/trans_dataset/clinc150/banking/target.train.jsonl')
    # parser.add_argument("--target-query-path", type=str, default='../../dataset/trans_dataset/SNIPS/target_query.jsonl')

    parser.add_argument("--data-path", type=str, default=None, help="Path to data (ARSC only)")
    parser.add_argument('--sup-path',type=str, default='../../dataset/trans_dataset/clinc150/banking/sup_data.jsonl', help="Path to sup data")

    parser.add_argument("--output-path", type=str, default=f'runs/{now()}')
    parser.add_argument("--model-name-or-path", type=str,
                        default='/opt/yfy/PythonProjects/DFEPN/models/DTN/save/bert_classifier_40.pkl',
                        # default='/opt/yfy/PythonProjects/DFEPN/language_modeling/transformer_models/clinc150/banking/fine-tuned/',
                        help="Transformer model to use")
    # parser.add_argument("--model-name-or-path", type=str,default='/opt/yfy/PythonProjects/Fewshot-SLU/data/bert/bert-base-uncased/', help="Transformer model to use")
    parser.add_argument("--n-unlabeled", type=int, help="Number of unlabeled data points per class (proto++)", default=0)
    parser.add_argument("--max-iter", type=int, default=400, help="Max number of training episodes")
    parser.add_argument("--evaluate-every", type=int, default=1, help="Number of training episodes between each evaluation (on both valid, test)")
    parser.add_argument("--log-every", type=int, default=1, help="Number of training episodes between each logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to set")
    parser.add_argument("--early-stop", type=int, default=0, help="Number of worse evaluation steps before stopping. 0=disabled")

    # Few-Shot related stuff
    parser.add_argument("--n-support", type=int, default=5, help="Number of support points for each class")
    parser.add_argument("--n-query", type=int, default=10, help="Number of query points for each class")
    parser.add_argument("--n-classes", type=int, default=15, help="Number of classes per episode")
    parser.add_argument("--n-test-episodes", type=int, default=1, help="Number of episodes during evaluation (valid, test)")

    # Metric to use in proto distance calculation
    parser.add_argument("--metric", type=str, default="euclidean", help="Metric to use", choices=("euclidean", "cosine"))

    # ARSC data
    parser.add_argument("--arsc-format", default=False, action="store_true", help="Using ARSC few-shot format")
    args = parser.parse_args()

    # Set random seed
    # set_seeds(args.seed)

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
        log_every=args.log_every
    )

    # Save config
    with open(os.path.join(args.output_path, "config.json"), "w") as file:
        json.dump(vars(args), file, ensure_ascii=False)


if __name__ == '__main__':
    main()
