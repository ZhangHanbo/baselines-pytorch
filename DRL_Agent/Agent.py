# father class for all agent
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
import Feature_Extractor
import abc
import copy
from config import AGENT_CONFIG
from utils import databuffer, databuffer_PG_gaussian

class Agent:
    __metaclass__ = abc.ABCMeta
    def __init__(self,hyperparams):
        config = copy.deepcopy(AGENT_CONFIG)
        config.update(hyperparams)
        self.n_states = config['n_states']
        self.n_actions = config['n_actions']
        self.lr = config['lr']
        self.mom = config['mom']
        self.gamma = config['reward_decay']
        self.memory_size = config['memory_size']
        self.batch_size = config['batch_size']
        self.learn_step_counter = 0
        self.cost_his = []

    @abc.abstractmethod
    def choose_action(self, s):
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def learn(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def store_transition(self, transition):
        # replace the old memory with new memory
        transition = torch.Tensor(transition)
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def sample_batch(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        return batch_memory, sample_index

    def soft_update(self, target, eval, tau):
        for target_param, param in zip(target.parameters(), eval.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) +
                                    param.data * tau)

    def hard_update(self, target, eval):
        target.load_state_dict(eval.state_dict())
        print('\ntarget_params_replaced\n')

