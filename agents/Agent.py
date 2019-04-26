# father class for all agent
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
import basenets
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

        # self.n_actions = 0
        if 'n_actions' in config.keys():
            self.n_actions = config['n_actions']

        self.n_action_dims = config['n_action_dims']
        self.lr = config['lr']
        self.mom = config['mom']
        self.gamma = config['reward_decay']
        self.memory_size = config['memory_size']

        self.max_num_episode = config['num_episode']
        self.max_num_step = config['num_step']
        self.snapshot_episode = config['snapshot_episode']
        self.update_num_per_step = config['update_num_per_step']
        self.hindsight_replay = config['hindsight_replay']

        self.learn_step_counter = 0
        self.episode_counter = 0
        self.cost_his = []

        self.use_cuda = False
        self.r = torch.FloatTensor(1)
        self.done = torch.FloatTensor(1)
        self.s_ = torch.FloatTensor(1)
        self.s = torch.FloatTensor(1)
        self.a = torch.FloatTensor(1)

    @abc.abstractmethod
    def choose_action(self, s):
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def learn(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def store_transition(self, transition):
        self.memory.store_transition(transition)

    def sample_batch(self, batch_size = None):
        return self.memory.sample_batch(batch_size)

    def soft_update(self, target, eval, tau):
        for target_param, param in zip(target.parameters(), eval.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) +
                                    param.data * tau)

    def hard_update(self, target, eval):
        target.load_state_dict(eval.state_dict())
        # print('\ntarget_params_replaced\n')

    def cuda(self):
        self.use_cuda = True
        self.r = self.r.cuda()
        self.a = self.a.cuda()
        self.s = self.s.cuda()
        self.s_ = self.s_.cuda()
        self.done = self.done.cuda()

    @abc.abstractmethod
    def save_model(self, save_path):
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def load_model(self, load_path, load_point):
        raise NotImplementedError("Must be implemented in subclass.")

