import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class IndRNNCell(nn.Module):
    def __init__(self, inpdim, recdim, act=None):
        super().__init__()
        self.inpdim = inpdim
        self.recdim = recdim
        self.act = F.reul if act is None else act
        self.w = nn.Parameter(th.randn(inpdim, recdim))
        self.u = nn.Parameter(th.randn(recdim))
        self.b = nn.Parameter(th.randn(recdim))
        self.F = nn.Linear(recdim, 1)

    def forward(self, x_t, h_tm1):
        return self.act(h_tm1 * self.u + x_t @ self.w + self.b)


class IndRNN(nn.Module):
    def __init__(self, inpdim, recdim):
        super().__init__()
        self.inpdim = inpdim
        self.recdim = recdim
        self.cell = IndRNN(inpdim, recdim)

    def forward(self, x):
        h_tm1 = Variable(th.ones(self.recdim))
        seq = []
        for i in range(x.size()[1]):
            x_t = x[:, i, :]
            h_tm1 = self.cell.forward(x_t, h_tm1)
            seq.append(h_tm1)
        return th.stack(seq, dim=1)
