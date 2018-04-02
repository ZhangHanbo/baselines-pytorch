import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
import Feature_Extractor
import copy
from DRL_Agent.Agent import Agent
from config import NAF_CONFIG
from ValueFunc.NAF import FCNAF

class NAF(Agent):
    def __init__(self,hyperparams):
        config = copy.deepcopy(NAF_CONFIG)
        config.update(hyperparams)
        super(NAF, self).__init__(config)
        self.action_bounds = config['action_bounds']
        self.noise = config['noise_var']
        self.exploration_noise_decrement = config['noise_decrease']
        self.noise_min = config['noise_min']
        self.replace_tau = config['tau']
        # initialize zero memory [s, a, r, s_]
        self.memory = torch.Tensor(np.zeros((self.memory_size, self.n_features * 2 + 2 + self.n_actions)))
        self.e_NAF = FCNAF(self.n_features, self.n_actions, action_active=F.tanh, action_scaler=self.action_bounds)
        self.t_NAF = FCNAF(self.n_features, self.n_actions, action_active=F.tanh, action_scaler=self.action_bounds)
        self.loss_func = config['loss']()
        self.optimizer = config['optimizer'](self.e_NAF.parameters(), lr=self.lr)

    def choose_action(self,s):
        s = Variable(torch.Tensor(s))
        anoise = torch.normal(torch.zeros(self.n_actions),self.noise * torch.ones(self.n_actions))
        _, preda,_ = self.e_NAF(s)
        return np.array(preda.data + anoise)

    def learn(self):
        # check to replace target parameters
        self.soft_update(self.t_NAF, self.e_NAF, self.replace_tau)
        # sample batch memory from all memory
        batch_memory = self.sample_batch()
        # update step
        r = Variable(batch_memory[:, self.n_features + self.n_actions:self.n_features + self.n_actions + 1])
        done = Variable(batch_memory[:, (self.n_features + self.n_actions +1):(self.n_features + self.n_actions+ 2)])
        s_ = Variable(batch_memory[:, -self.n_features:])
        V_, _, _ = self.t_NAF(s_)
        q_target = r + self.gamma * V_
        q_target = q_target.detach()

        a = Variable(batch_memory[:, self.n_features:(self.n_actions + self.n_features)])
        s = Variable(batch_memory[:, :self.n_features])
        V,mu,L = self.t_NAF(s)
        dist = a - mu
        A_list = []
        for t in range(self.batch_size):
            A_list.append(torch.sum(torch.pow(torch.matmul(dist[t],L[t]),2)))
        A = -0.5 * torch.cat(A_list).unsqueeze(1)
        q_eval = V + A

        self.e_NAF.zero_grad()
        self.loss = self.loss_func(q_eval, q_target)
        self.loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        self.noise = self.noise * (
                1 - self.exploration_noise_decrement) if self.noise > self.noise_min else self.noise_min