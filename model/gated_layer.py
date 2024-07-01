from .gate import *
from .layer import *

class LayerGated(nn.Module):
    def __init__(self, layer, gates):
        super(LayerGated, self).__init__()
        self.layer = layer
        self.gates = gates

    def forward(self, x, use_gates=True):
        if use_gates:
            return self.layer(x, self.gates.get_gates())
        else:
            return self.layer(x)

    def count_sparsity(self):
        '''
        return the number of parameters of sparse slices
        '''
        original_ranks = self.gates.ranks
        reduced_ranks = self.gates.gated_ranks()
        ret = self.layer.calc_dof(original_ranks) - self.layer.calc_dof(reduced_ranks)
        return ret


def fix_gate(model):
    for m in model.modules():
        if isinstance(m, SingleSTG):
            m.mu.requires_grad = False
            m.mu.grad = torch.zeros_like(m.mu)
            m.gate_fixed = True


def active_gate(model):
    for m in model.modules():
        if isinstance(m, SingleSTG):
            m.mu.requires_grad = True
            m.gate_fixed = False