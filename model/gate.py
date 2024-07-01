import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

minimal_rank = 1


class SingleSTG(nn.Module):
    def __init__(self, rank, sigma, fix=minimal_rank):
        super(SingleSTG, self).__init__()

        self.gate_fixed = False

        self.rank = rank
        self.mu = nn.Parameter(0.01*torch.randn(rank-fix))
        self.mu_fix = nn.Parameter(0.01*torch.randn(fix), requires_grad=False)


        self.sigma = sigma
        self.noise = nn.Parameter(torch.randn(rank))
        self.noise.requires_grad = False

    def hard_sigmoid(self, x):
        return torch.clamp(x+1.2, 0, 1)

    def gate(self):
        if self.gate_fixed:
            z = torch.cat((self.mu, self.mu_fix))
        else:
            z = torch.cat((self.mu, self.mu_fix)) + self.sigma*self.noise.normal_()*self.training
        gate = self.hard_sigmoid(z)

        return gate

    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return 0.5 * (1 + torch.erf(x / np.sqrt(2)))


class TTTR_STG(nn.Module):
    """
    Stochastic gates that can be paired with a TT/TR layer.
    """

    def __init__(self, ranks, sigma):
        super(TTTR_STG, self).__init__()
        self.ranks = ranks
        self.sigma = sigma
        self.gates = nn.ParameterList([SingleSTG(r, sigma) for r in ranks])

    def get_gates(self):
        return [gate.gate() for gate in self.gates]

    def gated_ranks(self):
        # count non-zero gates

        train = self.training
        if train:
            self.eval()
        ranks = [torch.sum(gate.gate() > 0).item() for gate in self.gates]
        if train:
            self.train()
        return ranks

    def regularizer(self):
        reg = []
        for gate in self.gates:
            reg.append(torch.mean(gate.regularizer((gate.mu+1.2)/self.sigma)))
        ret = torch.norm(torch.stack(reg), p=1)
        return ret

class SingleREINFORCEGate(nn.Module):
    def __init__(self, rank, fix=minimal_rank):
        super().__init__()

        self.gate_fixed = False

        self.rank = rank
        self.mu = nn.Parameter(0.01*torch.randn(rank-fix))
        self.mu_fix = nn.Parameter(0.01*torch.randn(fix), requires_grad=False)

    def hard_sigmoid(self, x):
        return torch.clamp(x+.55, 0, 1)

    def gate(self):
        z = torch.cat((self.mu, self.mu_fix))
        gate = self.hard_sigmoid(z)
        if self.training:
            gate = gate.bernoulli()
        else:
            # round to 0 or 1
            gate = torch.round(gate)
        return gate

    def regularizer(self):
        return torch.sum(self.gate() > 0).item()
        
    def logit_prob(self, gate):
        z = torch.cat((self.mu, self.mu_fix))
        z = self.hard_sigmoid(z)
        return torch.sum(gate * torch.log(z) + (1 - gate) * torch.log(1 - z))


class TTTR_REINFORCE_gate(nn.Module):
    def __init__(self, ranks) -> None:
        super().__init__()
        self.ranks = ranks
        self.sigma = 0
        self.gates = nn.ParameterList([SingleREINFORCEGate(r) for r in ranks])

        # record the last Bernoulli gates sampled
        self.samples = None

    def get_gates(self):
        self.samples = [gate.gate() for gate in self.gates]
        return self.samples
    
    def gated_ranks(self):
        return [torch.sum(gate.gate() > 0).item() for gate in self.gates]
    
    def regularizer(self):
        nz = 0
        if self.samples is None:
            raise ValueError('No samplings now.')

        for gate in self.samples:
            nz += torch.sum(gate > 0).item()
        return nz
    
    def logit_prob(self, gates=None):
        if gates is None:
            if self.samples is None:
                raise ValueError()
            else:
                gates = self.samples

        logit_prob = 0
        for gate, g in zip(self.gates, gates):
            logit_prob += gate.logit_prob(g)
        return logit_prob
