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
                 outactive = None,
                 outscaler = None
                 ):
        super(MLP,self).__init__()
        self.nonlinear = nonlinear
        self.outactive = outactive
        if outscaler is not None:
            if isinstance(outscaler, (int, long, float)):
                self.outscaler = Variable(torch.Tensor([outscaler]))
            else:
                self.outscaler = Variable(torch.Tensor(outscaler))
        else:
            self.outscaler = None
        inlists = np.hstack([n_inputfeats,n_hiddens])
        outlists = np.hstack([n_hiddens,n_outputfeats])
        self.layers = nn.ModuleList()
        for n_inunits, n_outunits in zip(inlists,outlists):
            self.layers.append(nn.Linear(n_inunits,n_outunits))
        for layer in self.layers:
            layer.weight.data.normal_(0, 0.1)

    def forward(self,x):
        for layernum, layer in enumerate(self.layers):
            x = layer(x)
            # the last layer
            if layernum == len(self.layers) -1 :
                if self.outactive is not None:
                    if self.outscaler is not None:
                        x = self.outscaler * self.outactive(x)
                    else:
                        x = self.outactive(x)
            else:
                x = self.nonlinear(x)
        return x