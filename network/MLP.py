import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self,
                 n_inputfeats,    # input dim
                 n_outputfeats,   # output dim
                 n_hiddens = [30],  # hidden unit number list
                 nonlinear = F.relu,
                 outactive = None
                 ):
        super(MLP,self).__init__()
        self.nonlinear = nonlinear
        self.outactive = outactive
        inlists = np.hstack([n_inputfeats,n_hiddens])
        outlists = np.hstack([n_hiddens,n_outputfeats])
        self.layers = nn.ModuleList()
        for n_inunits, n_outunits in zip(inlists,outlists):
            self.layers.append(nn.Linear(n_inunits,n_outunits))

    def forward(self,x):
        for layernum, layer in enumerate(self.layers):
            x = layer(x)
            if layernum == len(self.layers) -1 :
                if self.outactive:
                    x = self.outactive(x)
            else:
                x = self.nonlinear(x)
        return x
