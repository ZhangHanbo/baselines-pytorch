import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from basenets.MLP import MLP
from torch import nn

class FCPG_Gaussian(MLP):
    def __init__(self,
                 n_inputfeats,    # input dim
                 n_actions,   # output dim
                 sigma,
                 n_hiddens = [30],  # hidden unit number list
                 nonlinear = F.tanh,
                 usebn = False,
                 outactive = None,
                 outscaler = None,
                 initializer = "orthogonal",
                 initializer_param = {"gain":np.sqrt(2), "last_gain": 0.1}
                 ):
        self.n_actions = n_actions
        super(FCPG_Gaussian, self).__init__(
                 n_inputfeats,    # input dim
                 n_actions,   # output dim
                 n_hiddens,  # hidden unit number list
                 nonlinear,
                 usebn,
                 outactive,
                 outscaler,
                 initializer,
                 initializer_param=initializer_param,
                 )
        self.logstd = nn.Parameter(torch.log(sigma * torch.ones(n_actions) + 1e-8))

    def forward(self,x):
        x = MLP.forward(self, x)
        return x, self.logstd.expand_as(x), torch.exp(self.logstd).expand_as(x)

    def cuda(self, device = None):
        self.logstd.cuda()
        return self._apply(lambda t: t.cuda(device))

class FCPG_Softmax(MLP):
    def __init__(self,
                 n_inputfeats,    # input dim
                 n_actions,   # output dim
                 n_hiddens = [10],  # hidden unit number list
                 nonlinear = F.tanh,
                 usebn = False,
                 outactive = F.softmax,
                 outscaler = None,
                 initializer = "orthogonal",
                 initializer_param = {"gain":np.sqrt(2), "last_gain": 0.1}
                 ):
        self.n_actions = n_actions
        super(FCPG_Softmax, self).__init__(
                 n_inputfeats,    # input dim
                 n_actions,   # output dim
                 n_hiddens,  # hidden unit number list
                 nonlinear,
                 usebn,
                 outactive,
                 outscaler,
                 initializer,
                 initializer_param=initializer_param,
                 )

# TODO: support multi-layer value function in which action is concat before the final layer
class FCVALUE(MLP):
    def __init__(self,
                 n_inputfeats,
                 n_hiddens = [30],  # hidden unit number list
                 nonlinear = F.tanh,
                 usebn = False,
                 outactive = None,
                 outscaler = None,
                 initializer="orthogonal",
                 initializer_param={"gain":np.sqrt(2), "last_gain": 0.1}
                 ):
        super(FCVALUE, self).__init__(
                 n_inputfeats,    # input dim
                 1,   # output dim
                 n_hiddens,  # hidden unit number list
                 nonlinear,
                 usebn,
                 outactive,
                 outscaler,
                 initializer,
                 initializer_param=initializer_param,
                 )

