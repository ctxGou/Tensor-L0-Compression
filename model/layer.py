import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import tensorly as tl

# https://github.com/MaxBourdon/mars
class TensorizedModel(nn.Module):
    def __init__(self):
        "A tensorized model base class."
        super().__init__()
        self._ranks = None
        self._cores = None
        self._total = None

    def __str__(self):
        return "Abstract tensorized model"

    @property
    def cores(self):
        return self._cores

    @property
    def ranks(self):
        return self._ranks

    @property
    def total(self):
        return self._total

    def calc_dof(self, ranks=None):
        "Calculate degrees of freedom given ranks, i.e., the number of actually occupied parameters."
        raise NotImplementedError()


class TT_Linear2(TensorizedModel):
    '''
    TT-FC from Novikov, where 4-order cores hold 1 in-edge and 1 out-edge.
    '''

    def __init__(self, in_features, out_features, ranks, bias=True, use_gate=False):
        super(TT_Linear2, self).__init__()
        if len(in_features) != len(out_features):
            raise ValueError(
                'In features and out features length inconsistent!')
        self.in_features = in_features
        self.out_features = out_features
        self._ranks = ranks
        self.bias = bias
        var = 2**(1/len(self.in_features)) / \
            np.prod(np.power(ranks + in_features, 1/len(self.in_features)))
        ub = np.sqrt(3*var)

        self._cores = nn.ParameterList([
            nn.Parameter(torch.randn(
                ranks[i], in_features[i], out_features[i], ranks[i+1]).uniform_(-ub, ub))
            for i in range(len(self.in_features))
        ])
        if bias:
            self.bias = nn.Parameter(torch.randn(np.prod(out_features)))

        self._total = sum([np.prod(core.shape) for core in self._cores])

    def forward(self, x, masks=None):
        x = x.reshape(-1, *self.in_features)

        w = self._cores[0]
        for idx in range(1, len(self.in_features)):
            if masks is not None:
                w = w * masks[idx]
            core = self._cores[idx]
            w = torch.tensordot(w, core, dims=([-1], [0]))

        w_in_dims = list(range(1, 2*len(self.in_features)+1, 2))
        x_in_dims = list(range(1, len(self.in_features)+1))

        out = torch.tensordot(x, w, dims=(x_in_dims, w_in_dims))
        out = out.reshape(-1, np.prod(self.out_features))

        if self.bias is not None:
            out += self.bias

        return out

    def calc_dof(self, ranks=None):
        if ranks is None:
            ranks = self._ranks
        return sum([ranks[i]*self.in_features[i]*self.out_features[i]*ranks[i+1] for i in range(len(self.in_features))])

# https://github.com/tnbar/tednet/blob/main/tednet/tnn/tensor_train/base.py
class TT_Conv2d2(TensorizedModel):
    def __init__(self, in_features, out_features, ranks, kernel_size, stride=1, padding=0, bias=True):
        super(TT_Conv2d2, self).__init__()
        self.in_feature = np.prod(in_features)
        self.out_feature = np.prod(out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self._ranks = ranks[1:]
        self.bias = bias

        self.stride = stride
        self.padding = padding

        self.in_num = len(self.in_features)
        self.out_num = len(self.out_features)
        self.core_num = self.in_num

        n = self.in_num+1
        var = 2**(1/n)/np.prod(np.power(ranks + in_features + [kernel_size*kernel_size], 1/n))
        ub = np.sqrt(3*var)

        self._cores = nn.ParameterList([])
        for i in range(self.core_num):
            self._cores.append(torch.randn(self.in_features[i], self.out_features[i], self._ranks[i], self._ranks[i+1]).uniform_(-ub, ub))

        self.kernel_core = nn.Parameter(torch.randn(1, self._ranks[0], self.kernel_size, self.kernel_size).uniform_(-ub, ub))

        self._total = sum([np.prod(core.shape) for core in self._cores] + [np.prod(self.kernel.weight.shape)])
        if bias:
            self.bias = nn.Parameter(torch.randn(self.out_feature, 1, 1).uniform_(-ub, ub))
    
    def forward(self, x, masks=None):
        batch_size = x.shape[0]
        hw = x.shape[-2:]

        res = x.view(-1, 1, *hw)

        kernel = self.kernel_core * masks[1].reshape(-1, 1, 1, 1)

        res = F.conv2d(res, kernel, bias=None, stride=self.stride, padding=self.padding)
        new_hw = res.shape[-2:]

        res = res.view(batch_size, *self.in_features, self._ranks[0], -1)

        if masks is not None:
            res = torch.tensordot(res, self._cores[0]*masks[2], dims=([1, -2], [0, 2]))
        else:
            res = torch.tensordot(res, self._cores[0], dims=([1, -2], [0, 2]))

        for i in range(1, self.core_num):
            core = self._cores[i]
            if masks is not None:
                core = core * masks[i+2]
            res = torch.tensordot(res, core, dims=([1, -1], [0, 2]))
        res = res.reshape(batch_size, -1, *new_hw)
        
        if self.bias is not None:
            res += self.bias

        return res

    def calc_dof(self, ranks=None):
        if ranks is None:
            ranks = self._ranks
            ranks = [1]+ranks
        return sum([ranks[i+1]*self.in_features[i]*self.out_features[i]*ranks[i+2] for i in range(len(self.in_features))]) + self.kernel_size * self.kernel_size * ranks[0] * ranks[1]


class TR_Linear(TensorizedModel):
    def __init__(self, in_features, out_features, ranks, bias=True, use_gate=False, add_sparse=False):
        super(TR_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._ranks = ranks
        self.bias = bias
        self.dims = in_features + out_features
        var = 2**(1/len(self.dims)) / \
            np.prod(np.power(ranks + in_features, 1/len(self.dims)))
 
        ub = np.sqrt(3*var)

        ranks = [ranks[-1]] + ranks
        self._cores = nn.ParameterList([
            nn.Parameter(torch.randn(
                ranks[i], self.dims[i], ranks[i+1]).uniform_(-ub, ub))
            for i in range(len(self.dims))
        ])
        if bias:
            self.bias = nn.Parameter(torch.randn(np.prod(out_features)))

        self._total = sum([np.prod(core.shape) for core in self._cores])

        if add_sparse:
            self.weight_sparse = nn.Parameter(torch.randn(np.prod(out_features), np.prod(in_features)).zero_())
        self.add_sparse = add_sparse

    def forward(self, x, masks=None):
        w = self._cores[0]
        if masks is not None:
            w = w * masks[0]
        for i in range(1, len(self.in_features)):
            core = self._cores[i]
            if masks is not None:
                core = core * masks[i]
            w = torch.tensordot(w, core, dims=([-1], [0]))

        for i in range(len(self.in_features), len(self.in_features)+len(self.out_features)):
            core = self._cores[i]
            if masks is not None:
                core = core * masks[i]
            if i != len(self.in_features)+len(self.out_features)-1:
                w = torch.tensordot(w, core, dims=([-1], [0]))
            else:
                w = torch.tensordot(w, core, dims=([-1,0], [0,-1]))

        
        #print(w.shape)

        w = w.reshape(np.prod(self.out_features), -1)
        
        if self.add_sparse:
            w += self.weight_sparse

        return F.linear(x, w, self.bias)



    def calc_dof(self, ranks=None):
        if ranks is None:
            ranks = self._ranks
        ranks = [self._ranks[-1]] + ranks

        dof_tr = sum([ranks[i]*self.dims[i]*ranks[i+1] for i in range(len(self.dims))])


        if self.add_sparse:
            # count the number of parameters in weight_sparse
            # exclude the sparse rows and columns

            dof_sparse = np.sum(self.weight_sparse != 0)

        else:
            dof_sparse = 0

        return dof_tr + dof_sparse


class TR_Conv2d(TensorizedModel):
    def __init__(self, in_features, out_features, ranks, kernel_size, stride=1, padding=0, bias=True, use_gate=False, add_sparse=False):
        super(TR_Conv2d, self).__init__()
        if len(ranks) != len(in_features) + len(out_features) + 1:
            raise ValueError('Incorrect TRConv2d ranks!')

        self.in_feature = np.prod(in_features)
        self.out_feature = np.prod(out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self._ranks = ranks
        self.bias = bias
        self.dims = in_features + [kernel_size*kernel_size] + out_features

        self.stride = stride
        self.padding = padding

        # num of cores is len(self.dims)+1
        self.n_cores = len(self.dims) + 1
        var = np.power([self.in_feature, kernel_size**2] +
                       ranks, 1/self.n_cores)
        var = 2**(1/self.n_cores)/np.prod(var)

        ub = np.sqrt(3*var)

        ranks = [ranks[-1]] + ranks


        self._cores = nn.ParameterList([
            nn.Parameter(torch.randn(ranks[i], self.dims[i], ranks[i+1]).uniform_(-ub, ub))
            for i in range(len(self.in_features)+len(self.out_features)+1)
        ])

        self._total = sum([np.prod(core.shape)
                          for core in self._cores])

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_feature))
        else:
            self.bias = None

        if add_sparse:
            self.weight_sparse = nn.Parameter(torch.randn(self.out_feature, self.in_feature, kernel_size, kernel_size).zero_())
        self.add_sparse = add_sparse

    def forward(self, x, masks=None):
        factors = self._cores
        if masks is not None:
            factors = [factors[i]*masks[i] for i in range(len(factors))]
        weight = tl.tr_to_tensor(factors)
        weight = weight.reshape((self.out_feature, self.in_feature, self.kernel_size, self.kernel_size))
        if self.add_sparse:
            weight += self.weight_sparse
        return F.conv2d(x, weight, self.bias, stride=self.stride, padding=self.padding)

    def calc_dof(self, ranks=None):
        if ranks is None:
            ranks = self._ranks
        ranks = [self._ranks[-1]] + ranks

        dof_tr = sum([ranks[i]*self.dims[i]*ranks[i+1] for i in range(len(self.dims))])

        if self.add_sparse:
            dof_sparse = np.sum(self.weight_sparse != 0)
        else:
            dof_sparse = 0

        return dof_tr + dof_sparse
    
class TR_Embedding(TensorizedModel):
    def __init__(self, num_embeddings, embedding_dim, ranks):
        super(TR_Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._ranks = ranks
        self.dims = [num_embeddings, embedding_dim]
        var = 2**(1/len(self.dims)) / \
            np.prod(np.power(ranks + self.dims, 1/len(self.dims)))
        ub = np.sqrt(3*var)

        ranks = [ranks[-1]] + ranks
        self._cores = nn.ParameterList([
            nn.Parameter(torch.randn(
                ranks[i], self.dims[i], ranks[i+1]).uniform_(-ub, ub))
            for i in range(len(self.dims))
        ])

        self._total = sum([np.prod(core.shape) for core in self._cores])

    def forward(self, x, masks=None):
        factors = self._cores
        if masks is not None:
            factors = [factors[i]*masks[i] for i in range(len(factors))]
        weight = tl.tr_to_tensor(factors)
        return F.embedding(x, weight)

    def calc_dof(self, ranks=None):
        if ranks is None:
            ranks = self._ranks
        ranks = [self._ranks[-1]] + ranks
        dof_tr = sum([ranks[i]*self.dims[i]*ranks[i+1] for i in range(len(self.dims))])
        return dof_tr