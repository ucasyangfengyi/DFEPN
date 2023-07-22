import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


SEED = 3
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class GeneratorNet(nn.Module):
    def __init__(self,num_classes=80, norm=True, scale=True):
        super(GeneratorNet, self).__init__()
        # save_model = torch.load(model_name_or_path+'/pytorch_model.bin')

        self.add_info = AddInfo()
        self.generator = Generator()
        self.fc = nn.Linear(768, num_classes, bias=False)
        self.s = nn.Parameter(torch.FloatTensor([10]))

        # model_dict = self.add_info.state_dict()
        # state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        # model_dict.update(state_dict)
        # self.add_info.load_state_dict(model_dict)

    def forward(self, B1=None, B2=None, A=None, classifier=False):
        add_info = self.add_info(A, B1, B2)
        A_rebuild = self.generator(add_info)
        A_rebuild = self.l2_norm(A_rebuild)
        score = self.fc(A_rebuild*self.s)
        return A_rebuild, score
   
    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        norm = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(norm)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output


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
        # a = B1-B2
        out = A+(B1-B2)
        # out = B1-B2
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


if __name__ == '__main__':
    A = torch.ones((25,768))
    B1 = torch.ones((25,768))
    B2 = torch.ones((25,768))
    gen_net = GeneratorNet()
    gen_data, score = gen_net(A, B1, B2)
    print(gen_data.shape)
    print(score.shape)








