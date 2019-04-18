import torch
from basenets.MLP import MLP
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

class FCDQN(nn.Module):
    def __init__(self,
                 n_inputfeats,  # input dim
                 n_actions,  # action dim
                 n_hiddens=[30],  # hidden unit number list
                 nonlinear=F.relu
                 ):
        super(FCDQN, self).__init__()
        self.net = MLP(n_inputfeats,
                        n_actions,
                        n_hiddens = n_hiddens,
                        nonlinear = nonlinear)

    def forward(self, x):
        return self.net(x)

class FCDuelingDQN(nn.Module):
    def __init__(self,
                 n_inputfeats,
                 n_actions,
                 n_hiddens=[30],
                 nonlinear=F.relu):
        super(FCDuelingDQN, self).__init__()
        # using MLP as hidden layers
        self.hidden_layers = MLP(
            n_inputfeats,
            n_hiddens[-1],
            n_hiddens[:-1],
            nonlinear,
            outactive = nonlinear
        )
        self.V = nn.Linear(n_hiddens[-1], 1)
        self.A = nn.Linear(n_hiddens[-1], n_actions)
        self.V.weight.data.normal_(0, 0.1)
        self.A.weight.data.normal_(0, 0.1)

    def forward(self,x):
        x = self.hidden_layers.forward(x)
        A = self.A(x)-torch.mean(self.A(x),1,keepdim=True)
        V = self.V(x)
        return A+V