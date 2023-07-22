import sys
sys.path.append('/opt/yfy/PythonProjects/DFEPN')

import torch
import torch.nn as nn
import numpy as np
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