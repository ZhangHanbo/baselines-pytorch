import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from actors.MLP import MLP
from torch import nn

class FCPG_Gaussian(MLP):
    def __init__(self,
                 n_inputfeats,    # input dim
                 n_actions,   # output dim
                 sigma,
                 n_hiddens = [30],  # hidden unit number list
                 nonlinear = F.relu,
                 outactive = None,
                 outscaler = None
                 ):
        self.n_actions = n_actions
        super(FCPG_Gaussian, self).__init__(
                 n_inputfeats,    # input dim
                 n_actions,   # output dim
                 n_hiddens,  # hidden unit number list
                 nonlinear,
                 outactive,
                 outscaler
                 )
        self.sigma = nn.Parameter(sigma * torch.ones(n_actions))

    def forward(self,x):
        x = MLP.forward(self, x)
        return x, torch.pow(self.sigma.expand_as(x),2)

class FCPG_Softmax(MLP):
    def __init__(self,
                 n_inputfeats,    # input dim
                 n_actions,   # output dim
                 n_hiddens = [10],  # hidden unit number list
                 nonlinear = F.relu,
                 outactive = F.softmax,
                 outscaler = None
                 ):
        self.n_actions = n_actions
        super(FCPG_Softmax, self).__init__(
                 n_inputfeats,    # input dim
                 n_actions,   # output dim
                 n_hiddens,  # hidden unit number list
                 nonlinear,
                 outactive,
                 outscaler
                 )
        for i,v in enumerate(self.layers):
            torch.nn.init.normal(v.weight, mean = 0, std = 0.3)
            torch.nn.init.constant(v.bias, 0.1)

# TODO: support multi-layer value function in which action is concat before the final layer
class FCVALUE(MLP):
    def __init__(self,
                 n_inputfeats,
                 n_hiddens = [30],  # hidden unit number list
                 nonlinear = F.relu,
                 outactive = None,
                 outscaler = None
                 ):
        super(FCVALUE, self).__init__(
                 n_inputfeats,    # input dim
                 1,   # output dim
                 n_hiddens,  # hidden unit number list
                 nonlinear,
                 outactive,
                 outscaler
                 )

