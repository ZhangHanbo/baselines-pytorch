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
                 usebn = False,
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
        inlists = [n_inputfeats,] + n_hiddens
        outlists = n_hiddens + [n_outputfeats,]
        self.layers = nn.ModuleList()
        for n_inunits, n_outunits in zip(inlists,outlists):
            if usebn:
                bn_layer = nn.BatchNorm1d(n_inunits)
                bn_layer.weight.data.fill_(1)
                bn_layer.bias.data.fill_(0)
                self.layers.append(bn_layer)
            linear_layer = nn.Linear(n_inunits,n_outunits)
            linear_layer.weight.data.normal_(0, 0.1)
            self.layers.append(linear_layer)

    def forward(self,x):
        input_dim = x.dim()
        if input_dim == 1:
            x = x.unsqueeze(0)
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
        if input_dim == 1:
            x = x.squeeze(0)
        return x