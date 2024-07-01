import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from .layer import TT_Linear2, TT_Conv2d2, TR_Linear, TR_Conv2d
from .gate import TTTR_STG, TTTR_REINFORCE_gate
from .gated_layer import LayerGated, fix_gate, active_gate



class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.fc1 = nn.Linear(1250, 320)
        self.fc2 = nn.Linear(320, 10)

    def forward(self, x, use_gates=True):
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)

        out = out.view(x.shape[0], -1)

        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out

    def count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def structure(self):
        return None

    def regularizer(self):
        return torch.zeros([])

class LeNet5_TT2(nn.Module):
    def __init__(self, layer_ranks=[20,20,20,20]):
        super(LeNet5_TT2, self).__init__()
        r = layer_ranks
        self.conv1 = TT_Conv2d2([1,1], [4,5], [1,r[0],r[0],1],5,padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv2 = TT_Conv2d2([4,5], [5,10], [1,r[1],r[1],1],5)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.fc1 = TT_Linear2([5,5,5,10],[4,4,4,5], [1,r[2],r[2],r[2],1])
        self.fc2 = TT_Linear2([4,4,4,5], [1,1,1,10], [1,r[3],r[3],r[3],1])

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)

        out = out.view(x.shape[0], -1)

        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out

    def count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def structure(self):
        return None

    def regularizer(self):
        return torch.zeros([])

class LeNet5_TR(nn.Module):
    def __init__(self, layer_ranks=[20,20,20,20]):
        super(LeNet5_TR, self).__init__()
        r = layer_ranks
        self.conv1 = TR_Conv2d([1], [4,5], [r[0],r[0],r[0],r[0]],5,padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv2 = TR_Conv2d([4,5], [5,10], [r[1],r[1],r[1],r[1],r[1]],5)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.fc1 = TR_Linear([5,5,5,10],[5,8,8], [r[2],r[2],r[2],r[2],r[2],r[2],r[2]])
        self.fc2 = TR_Linear([5,8,8],[10],[r[3],r[3],r[3],r[3]])

    def forward(self, x, use_gates=True):
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)

        out = out.view(x.shape[0], -1)

        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out

    def count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def structure(self):
        return None

    def regularizer(self):
        return torch.zeros([])

class LeNet5_TT2_STG(nn.Module):
    def __init__(self, layer_ranks=[20,20,20,20], sigma=.5):
        super(LeNet5_TT2_STG, self).__init__()
        r = layer_ranks
        self.conv1 = LayerGated(TT_Conv2d2([1,1], [4,5], [1,r[0],r[0],1],5,padding=2), TTTR_STG([1,r[0],r[0],1], sigma))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv2 = LayerGated(TT_Conv2d2([4,5], [5,10], [1,r[1],r[1],1],5), TTTR_STG([1,r[1],r[1],1], sigma))
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.fc1 = LayerGated(TT_Linear2([5,5,5,10],[4,4,4,5], [1,r[2],r[2],r[2],1]), TTTR_STG([1,r[2],r[2],r[2],1], sigma))
        self.fc2 = LayerGated(TT_Linear2([4,4,4,5], [1,1,1,10], [1,r[3],r[3],r[3],1]), TTTR_STG([1,r[3],r[3],r[3],1], sigma))

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)

        out = out.view(x.shape[0], -1)

        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out

    def count(self):
        sparsity = self.conv1.count_sparsity() + self.conv2.count_sparsity() + self.fc1.count_sparsity() + self.fc2.count_sparsity()
        return sum(p.numel() for p in self.parameters() if p.requires_grad) - sparsity

    def structure(self):
        ret = [self.conv1.gates.gated_ranks(), self.conv2.gates.gated_ranks(), self.fc1.gates.gated_ranks(), self.fc2.gates.gated_ranks()]
        return ret

    def regularizer(self):
        return self.conv1.gates.regularizer() + self.conv2.gates.regularizer() + self.fc1.gates.regularizer() + self.fc2.gates.regularizer()

class LeNet5_TR_STG(nn.Module):
    def __init__(self, layer_ranks=[20,20,20,20], sigma=.5):
        super(LeNet5_TR_STG, self).__init__()

        r = layer_ranks
        self.conv1 = LayerGated(TR_Conv2d([1], [4,5], [r[0],r[0],r[0],r[0]],5,padding=2, use_gate=True), TTTR_STG([r[0],r[0],r[0],r[0]], sigma))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv2 = LayerGated(TR_Conv2d([4,5], [5,10], [r[1],r[1],r[1],r[1],r[1]],5, use_gate=True), TTTR_STG([r[1],r[1],r[1],r[1],r[1]], sigma))
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.fc1 = LayerGated(TR_Linear([5,5,5,10],[5,8,8], [r[2],r[2],r[2],r[2],r[2],r[2],r[2]], use_gate=True), TTTR_STG([r[2],r[2],r[2],r[2],r[2],r[2],r[2]], sigma))
        self.fc2 = LayerGated(TR_Linear([5,8,8], [10], [r[3],r[3],r[3],r[3]], use_gate=True), TTTR_STG([r[3],r[3],r[3],r[3]], sigma))

    def forward(self, x, use_gates=True):
        if use_gates:
            out = F.relu(self.conv1(x))
            out = self.pool1(out)
            out = F.relu(self.conv2(out))
            out = self.pool2(out)

            out = out.view(x.shape[0], -1)

            out = F.relu(self.fc1(out))
            out = self.fc2(out)

            return out
        else:
            out = F.relu(self.conv1(x, use_gates=False))
            out = self.pool1(out)
            out = F.relu(self.conv2(out, use_gates=False))
            out = self.pool2(out)

            out = out.view(x.shape[0], -1)

            out = F.relu(self.fc1(out, use_gates=False))
            out = self.fc2(out, use_gates=False)

            return out

    def count(self):
        sparsity = self.conv1.count_sparsity() + self.conv2.count_sparsity() + self.fc1.count_sparsity() + self.fc2.count_sparsity()
        return sum(p.numel() for p in self.parameters() if p.requires_grad) - sparsity

    def structure(self):
        ret = [self.conv1.gates.gated_ranks(), self.conv2.gates.gated_ranks(), self.fc1.gates.gated_ranks(), self.fc2.gates.gated_ranks()]
        return ret

    def regularizer(self):
        reg = [self.conv1.gates.regularizer(), self.conv2.gates.regularizer(), self.fc1.gates.regularizer(), self.fc2.gates.regularizer()]
        reg = sum(reg)
        return reg

class LeNet5_TR_REINFORCE(nn.Module):
    def __init__(self, layer_ranks=[20,20,20,20]):
        super(LeNet5_TR_REINFORCE, self).__init__()
        r = layer_ranks
        self.conv1 = LayerGated(TR_Conv2d([1], [4,5], [r[0],r[0],r[0],r[0]],5,padding=2, use_gate=True), TTTR_REINFORCE_gate([r[0],r[0],r[0],r[0]]))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv2 = LayerGated(TR_Conv2d([4,5], [5,10], [r[1],r[1],r[1],r[1],r[1]],5, use_gate=True), TTTR_REINFORCE_gate([r[1],r[1],r[1],r[1],r[1]]))
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.fc1 = LayerGated(TR_Linear([5,5,5,10],[5,8,8], [r[2],r[2],r[2],r[2],r[2],r[2],r[2]], use_gate=True), TTTR_REINFORCE_gate([r[2],r[2],r[2],r[2],r[2],r[2],r[2]]))
        self.fc2 = LayerGated(TR_Linear([5,8,8], [10], [r[3],r[3],r[3],r[3]], use_gate=True), TTTR_REINFORCE_gate([r[3],r[3],r[3],r[3]]))

    def forward(self, x):

        

        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)

        out = out.view(x.shape[0], -1)

        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out

    def count(self):
        sparsity = self.conv1.count_sparsity() + self.conv2.count_sparsity() + self.fc1.count_sparsity() + self.fc2.count_sparsity()
        return sum(p.numel() for p in self.parameters() if p.requires_grad) - sparsity
    
    def structure(self):
        ret = [self.conv1.gates.gated_ranks(), self.conv2.gates.gated_ranks(), self.fc1.gates.gated_ranks(), self.fc2.gates.gated_ranks()]
        return ret
    
    def regularizer(self):
        reg = [self.conv1.gates.regularizer(), self.conv2.gates.regularizer(), self.fc1.gates.regularizer(), self.fc2.gates.regularizer()]
        reg = sum(reg)
        return reg
    
    def logit_prob(self):
        logit_prob = self.conv1.gates.logit_prob() + self.conv2.gates.logit_prob() + self.fc1.gates.logit_prob() + self.fc2.gates.logit_prob()
        return logit_prob


def get_lenet5_model(name, sigma=None):
    if name == 'lenet5':
        return LeNet5()
    elif name == 'lenet5_tt2_25':
        return LeNet5_TT2(layer_ranks=[25,25,25,25])
    elif name == 'lenet5_tt2_15':
        return LeNet5_TT2(layer_ranks=[15,15,15,15])
    elif name == 'lenet5_tt2_20':
        return LeNet5_TT2(layer_ranks=[20,20,20,20])
    elif name == 'lenet5_tt2_10':
        return LeNet5_TT2(layer_ranks=[10,10,10,10])
    elif name == 'lenet5_tt2_25_stg':
        return LeNet5_TT2_STG(layer_ranks=[25,25,25,25], sigma=sigma)
    elif name == 'lenet5_tt2_15_stg':
        return LeNet5_TT2_STG(layer_ranks=[15,15,15,15], sigma=sigma)
    elif name == 'lenet5_tt2_20_stg':
        return LeNet5_TT2_STG(layer_ranks=[20,20,20,20], sigma=sigma)
    elif name == 'lenet5_tt2_10_stg':
        return LeNet5_TT2_STG(layer_ranks=[10,10,10,10], sigma=sigma)
    elif name == 'lenet5_tr_25':
        return LeNet5_TR(layer_ranks=[25,25,25,25])
    elif name == 'lenet5_tr_15':
        return LeNet5_TR(layer_ranks=[15,15,15,15])
    elif name == 'lenet5_tr_20':
        return LeNet5_TR(layer_ranks=[20,20,20,20])
    elif name == 'lenet5_tr_10':
        return LeNet5_TR(layer_ranks=[10,10,10,10])
    elif name == 'lenet5_tr_5':
        return LeNet5_TR(layer_ranks=[5,5,5,5])
    elif name == 'lenet5_tr_3':
        return LeNet5_TR(layer_ranks=[3,3,3,3])
    elif name == 'lenet5_tr_25_stg':
        return LeNet5_TR_STG(layer_ranks=[25,25,25,25], sigma=sigma)
    elif name == 'lenet5_tr_15_stg':
        return LeNet5_TR_STG(layer_ranks=[15,15,15,15], sigma=sigma)
    elif name == 'lenet5_tr_20_stg':
        return LeNet5_TR_STG(layer_ranks=[20,20,20,20], sigma=sigma)
    elif name == 'lenet5_tr_10_stg':
        return LeNet5_TR_STG(layer_ranks=[10,10,10,10], sigma=sigma)
    elif name == 'lenet5_tr_25_reinforce':
        return LeNet5_TR_REINFORCE(layer_ranks=[25,25,25,25])
