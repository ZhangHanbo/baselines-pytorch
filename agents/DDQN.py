import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
import basenets
from agents.DQN import DQN
from rlnets import FCDQN
import copy
from config import DQN_CONFIG

class DDQN(DQN):
    def __init__(self,hyperparams):
        config = copy.deepcopy(DQN_CONFIG)
        config.update(hyperparams)
        DQN.__init__(self, config)
        if type(self) == DDQN:
            self.e_DQN = FCDQN(self.n_states, self.n_actions,
                               n_hiddens=config['hidden_layers'],
                               usebn=config['use_batch_norm'],
                               nonlinear=config['act_func'])
            self.t_DQN = FCDQN(self.n_states, self.n_actions,
                               n_hiddens=config['hidden_layers'],
                               usebn=config['use_batch_norm'],
                               nonlinear=config['act_func'])
            self.lossfunc = config['loss']()
            if self.mom == 0 or self.mom is None:
                self.optimizer = config['optimizer'](self.e_DQN.parameters(),lr = self.lr)
            else:
                self.optimizer = config['optimizer'](self.e_DQN.parameters(), lr=self.lr, momentum = self.mom)

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.hard_update(self.t_DQN, self.e_DQN)
        batch_memory = self.sample_batch(self.batch_size)[0]

        r = torch.Tensor(batch_memory['reward'])
        s_ = torch.Tensor(batch_memory['next_state'])
        q_target = self.t_DQN(s_)
        q_eval_wrt_s_ = self.e_DQN(s_)
        a_eval_wrt_s_ = torch.max(q_eval_wrt_s_,1)[1].view(self.batch_size, 1)
        q_target = r + self.gamma * q_target.gather(1, a_eval_wrt_s_)

        s = torch.Tensor(batch_memory['state'])
        q_eval = self.e_DQN(s)
        q_eval_wrt_a = q_eval.gather(1, torch.LongTensor(batch_memory['action']))
        q_target = q_target.detach()

        self.loss = self.lossfunc(q_eval_wrt_a, q_target)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.cost_his.append(self.loss.data)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
