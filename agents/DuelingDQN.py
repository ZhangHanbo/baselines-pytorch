import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
import basenets
from agents.DQN import DQN
from agents.DDQN import DDQN
import copy
from config import DQN_CONFIG
from critics import FCDuelingDQN

class DuelingDQN(DDQN):
    def __init__(self,hyperparams):
        config = copy.deepcopy(DQN_CONFIG)
        config.update(hyperparams)
        DQN.__init__(self, config)
        self.e_DQN = FCDuelingDQN(self.n_states, self.n_actions, n_hiddens = [30])
        self.t_DQN = FCDuelingDQN(self.n_states, self.n_actions, n_hiddens = [30])
        self.lossfunc = config['loss']()
        if self.mom == 0 or self.mom is None:
            self.optimizer = config['optimizer'](self.e_DQN.parameters(), lr=self.lr)
        else:
            self.optimizer = config['optimizer'](self.e_DQN.parameters(), lr=self.lr, momentum=self.mom)
