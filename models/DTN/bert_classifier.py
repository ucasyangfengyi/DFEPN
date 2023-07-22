import sys
sys.path.append('/opt/yfy/PythonProjects/DFEPN')
from typing import List
import torch.nn as nn
import logging
import warnings
import torch
from transformers import BertTokenizer,BertModel,BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from utils.data import get_jsonl_data
import tqdm
from utils.python import now, set_seeds
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.simplefilter('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device('cpu')


class BertClassifier(nn.Module):
    def __init__(self, config_name_or_path,num_labels=135):
        super(BertClassifier, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(config_name_or_path)
        self.bert = BertModel.from_pretrained(config_name_or_path)
        self.num_labels = num_labels
        self.classifier = nn.Linear(768, num_labels,bias=True)

    def forward(self, sentences: List[str]):
        # batch_size = 16
        # if len(sentences) > batch_size:
        #     return torch.cat([self.forward(sentences[i:i + batch_size]) for i in range(0, len(sentences), batch_size)], 0)
        encoded_plus = [self.tokenizer.encode_plus(s, max_length=128,truncation=True) for s in sentences]
        # max_len = max([len(e['input_ids']) for e in encoded_plus])
        max_len = 30
        input_ids = list()
        attention_masks = list()
        token_type_ids = list()

        for e in encoded_plus:
            e['input_ids'] = e['input_ids'][:max_len]
            e['token_type_ids'] = e['token_type_ids'][:max_len]
            pad_len = max_len - len(e['input_ids'])
            input_ids.append(e['input_ids'] + pad_len * [self.tokenizer.pad_token_id])
            attention_masks.append([1 for _ in e['input_ids']] + [0] * pad_len)
            token_type_ids.append(e['token_type_ids'] + [0] * pad_len)

        outputs = self.bert.forward(input_ids=torch.Tensor(input_ids).long().to(device),
                                 attention_mask=torch.Tensor(attention_masks).long().to(device),
                                 token_type_ids=torch.Tensor(token_type_ids).long().to(device))
        pooled_output = outputs[1]
        sequence_output = outputs[0]
        logits = self.classifier(pooled_output)
        return logits,pooled_output,sequence_output

class BaseTrainDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, train_jsonl):
        self.train_data = get_jsonl_data(train_jsonl)
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
        for data in self.train_data:
            sentence_label = data['label']
            if sentence_label not in label2id:
                label2id[sentence_label] = id
                id2label[id] = sentence_label
                id += 1
            sentence = data['sentence']
            label = label2id[sentence_label]
            sentence_list.append(sentence)
            label_list.append(label)
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
        id = 0
        for data in self.train_data:
            sentence_label = data['label']
            sentence = data['sentence']
            label = self.label2id[sentence_label]
            sentence_list.append(sentence)
            label_list.append(label)
        return sentence_list,label_list

def base_train(base_train_dataloader,bert_classifier,base_dev_dataloader,
               optimizer,criterion,output_path,
               epochs=10,num_labels=135,log_every=10):
    best_base_acc = 0
    for epoch in tqdm.tqdm(range(epochs)):
        count = 0
        for data in tqdm.tqdm(base_train_dataloader):
            count += 1
            bert_classifier.train()
            optimizer.zero_grad()
            sentences = data['sentences']
            labels = data['labels'].to(device)
            logits, _ = bert_classifier(sentences)
            loss = criterion(logits.view(-1, num_labels), labels.view(-1))
            predict = torch.max(logits, 1)[1]
            correct = torch.eq(predict, labels).sum().float().item()
            acc = correct / len(labels)
            loss.backward()
            optimizer.step()
            if count % log_every == 0:
                logger.info(f"BertClassifier train | loss: {loss:.4f} | acc: {acc:.4f}")
        torch.save(bert_classifier.state_dict(), output_path + '/bert_classifier_20.pkl')
        logger.info(f"The best bertclassifier model saved")

        # correct = 0
        # len_dev = 0
        # for data in tqdm.tqdm(base_dev_dataloader):
        #     sentences = data['sentences']
        #     labels = data['labels'].to(device)
        #     logits, _ = bert_classifier(sentences)
        #     loss = criterion(logits.view(-1, num_labels), labels.view(-1))
        #     predict = torch.max(logits, 1)[1]
        #     correct += torch.eq(predict, labels).sum().float().item()
        #     len_dev += len(labels)
        # acc = correct / len_dev
        # logger.info(f"BertClassifier dev | loss: {loss:.4f} | acc: {acc:.4f}")
        # if acc > best_base_acc:
        #     best_base_acc = acc
        #     torch.save(bert_classifier.state_dict(), output_path + '/bert_classifier_135.pkl')
        #     logger.info(f"The best bertclassifier model saved")


if __name__ == '__main__':
    set_seeds(5)
    num_labels = 20
    log_every = 10
    epochs = 1
    output_path = 'save'
    # bert_path = '/opt/yfy/PythonProjects/Fewshot-SLU/data/bert/bert-base-uncased/'
    # bert_path = '/opt/yfy/PythonProjects/DFEPN/language_modeling/transformer_models/clinc150/banking/fine-tuned/'
    nlue_id = 1
    bert_path = '/opt/yfy/PythonProjects/DFEPN/language_modeling/transformer_models/nlue/kfold_' \
                         + str(nlue_id) + '/fine-tuned/'
    bert_classifier = BertClassifier(bert_path,num_labels).to(device)
    sentence = 'hello world'
    x = bert_classifier(sentence)
    print()

    # criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.Adam([
    #     {'params': bert_classifier.bert.parameters(),'lr':2e-5},
    #     {'params': bert_classifier.classifier.parameters(), 'lr': 1e-4}
    #     ]
    # )
    # base_train_dataset = BaseTrainDataset('../../dataset/trans_dataset/NLUE/KFold_1/auxiliary_samples.jsonl')
    # base_train_dataloader = DataLoader(base_train_dataset, batch_size=16, shuffle=True)
    #
    # base_dev_dataset = BaseTrainDataset('../../dataset/trans_dataset/clinc150/banking/dev.jsonl')
    # base_dev_dataloader = DataLoader(base_dev_dataset, batch_size=16, shuffle=True)
    #
    # base_train(base_train_dataloader, bert_classifier, base_dev_dataloader,
    #            optimizer, criterion, output_path,
    #            epochs, num_labels, log_every)



    # target_dev_dataset = BaseTrainDataset('../../dataset/trans_dataset/clinc150/banking/target.test.jsonl')
    # target_dev_dataloader = DataLoader(target_dev_dataset, batch_size=10, shuffle=False)
    # bert_classifier.load_state_dict(torch.load(output_path + '/bert_classifier_15_finetune.pkl'))
    # save_model = torch.load(output_path + '/bert_classifier_15_finetune.pkl')
    # W_base = save_model['classifier.weight']
    # W_base = bert_classifier.classifier.weight.data
    # print()
    # model_dict = mynetwork.state_dict()
    # state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
    # model_dict.update(state_dict)
    # mynetwork.load_state_dict(model_dict)

    # correct = 0
    # len_dev = 0
    # for data in target_dev_dataloader:
    #     sentences = data['sentences']
    #     labels = data['labels'].to(device)
    #     logits, _ = bert_classifier(sentences)
    #     loss = criterion(logits.view(-1, num_labels), labels.view(-1))
    #     predict = torch.max(logits, 1)[1]
    #     print(predict)
    #     correct += torch.eq(predict, labels).sum().float().item()
    #     len_dev += len(labels)
    # acc = correct / len_dev
    # logger.info(f"BertClassifier dev | loss: {loss:.4f} | acc: {acc:.4f}")

    # for data in tqdm.tqdm(target_dev_dataloader):
    #     sentences = data['sentences']
    #     labels = data['labels'].to(device)
    #     logits, embedding = bert_classifier(sentences)
    #     print()

#     下午尝试将target的数据输入，看一下情况,下午写meta train部分的代码




