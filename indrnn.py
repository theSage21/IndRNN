import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class IndRNNCell(nn.Module):
    """
    IndRNN Cell

    Performs a single time step operation

    """
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
    """
    IndRNN

    Given an input sequence, converts it to an output sequence.
    """
    def __init__(self, inpdim, recdim, depth=1):
        """
        inpdim      : dimension D in (Batch, Time, D)
        recdim      : recurrent dimension/ Units/
        depth       : stack depth
        """
        super().__init__()
        self.inpdim = inpdim
        self.recdim = recdim
        self.cells = [IndRNN(inpdim, recdim)
                      for _ in range(depth)]
        self.depth = depth

    def forward(self, x):
        h_tm1 = Variable(th.ones(self.recdim))
        seq = []
        for i in range(x.size()[1]):
            x_t = x[:, i, :]
            for cell in self.cells:
                h_tm1 = cell.forward(x_t, h_tm1)
            seq.append(h_tm1)
        return th.stack(seq, dim=1)
