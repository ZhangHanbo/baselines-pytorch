import torch
from Feature_Extractor.MLP import MLP
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

class FCDQN(MLP):
    def __init__(self,
                 n_inputfeats,  # input dim
                 n_actions,  # action dim
                 n_hiddens=[30],  # hidden unit number list
                 nonlinear=F.relu
                 ):
        super(FCDQN, self).__init__(n_inputfeats,
                                    n_actions,
                                    n_hiddens = n_hiddens,
                                    nonlinear = nonlinear)