import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from Feature_Extractor.MLP import MLP

class FCPOLICYTRPO(MLP):
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
        if sigma is not None:
            self.fixedsigma = True
            super(FCPOLICYTRPO, self).__init__(
                 n_inputfeats,    # input dim
                 n_actions,   # output dim
                 n_hiddens,  # hidden unit number list
                 nonlinear,
                 outactive,
                 outscaler
                 )
            if isinstance(sigma,(int, long, float)):
                self.sigma = sigma * Variable(torch.ones(n_actions))
            else:
                self.sigma = Variable(torch.Tensor(sigma))
        else:
            self.fixedsigma = False
            super(FCPOLICYTRPO, self).__init__(
                 n_inputfeats,    # input dim
                 n_actions * 2,   # output dim
                 n_hiddens,  # hidden unit number list
                 nonlinear,
                 outactive,
                 outscaler
                 )
    def forward(self,x):
        if self.fixedsigma:
            x = MLP.forward(self, x)
            return x, self.sigma.expand_as(x)
        else:
            x = MLP.forward(self,x)
            if x.dim() == 1:
                return x[:self.n_actions], x[-self.n_actions:]
            else:
                return x[:,:self.n_actions],x[:,-self.n_actions:]

# TODO: support multi-layer value function in which action is concat before the final layer
class FCVALUETRPO(MLP):
    def __init__(self,
                 n_inputfeats,
                 n_hiddens = [30],  # hidden unit number list
                 nonlinear = F.relu,
                 outactive = None,
                 outscaler = None
                 ):
        super(FCVALUETRPO, self).__init__(
                 n_inputfeats,    # input dim
                 1,   # output dim
                 n_hiddens,  # hidden unit number list
                 nonlinear,
                 outactive,
                 outscaler
                 )
        def forward(self,s,a):
            return MLP.forward(self,torch.cat([s,a],dim = 1))