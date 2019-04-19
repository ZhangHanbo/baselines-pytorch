import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
import basenets
import copy
from agents.Agent import Agent
from config import NAF_CONFIG
from critics.NAF import FCNAF
from utils import databuffer

class NAF(Agent):
    def __init__(self,hyperparams):
        config = copy.deepcopy(NAF_CONFIG)
        config.update(hyperparams)
        super(NAF, self).__init__(config)
        self.batch_size = config['batch_size']
        self.action_bounds = config['action_bounds']
        self.noise = config['noise_var']
        self.exploration_noise_decrement = config['noise_decrease']
        self.noise_min = config['noise_min']
        self.replace_tau = config['tau']
        # initialize zero memory [s, a, r, s_]
        self.memory = databuffer(config)
        self.e_NAF = FCNAF(self.n_states, self.n_action_dims, action_active=F.tanh, action_scaler=self.action_bounds)
        self.t_NAF = FCNAF(self.n_states, self.n_action_dims, action_active=F.tanh, action_scaler=self.action_bounds)
        self.loss_func = config['loss']()
        self.optimizer = config['optimizer'](self.e_NAF.parameters(), lr=self.lr)

    def choose_action(self,s):
        s = torch.Tensor(s)
        anoise = torch.normal(torch.zeros(self.n_action_dims),self.noise * torch.ones(self.n_action_dims))
        _, preda,_ = self.e_NAF(s)
        return np.array(preda.data + anoise)

    def learn(self):
        # check to replace target parameters
        self.soft_update(self.t_NAF, self.e_NAF, self.replace_tau)

        # sample batch memory from all memory
        batch_memory = self.sample_batch(self.batch_size)[0]
        r = torch.Tensor(batch_memory['reward'])
        done = torch.Tensor(batch_memory['done'])
        s_ = torch.Tensor(batch_memory['next_state'])
        a = torch.Tensor(batch_memory['action'])
        s = torch.Tensor(batch_memory['state'])

        V_, _, _ = self.t_NAF(s_)
        q_target = r + self.gamma * V_
        q_target = q_target.squeeze().detach()

        V,mu,L = self.e_NAF(s)
        a_mu = a - mu
        a_muxL = torch.bmm(a_mu.unsqueeze(1),L)
        A = -0.5 * torch.bmm( a_muxL, a_muxL.transpose(1,2)).squeeze()
        q_eval = V.squeeze() + A

        self.e_NAF.zero_grad()
        self.loss = self.loss_func(q_eval, q_target)
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.e_NAF.parameters(), 1)
        self.optimizer.step()

        self.learn_step_counter += 1
        self.noise = self.noise * (
                1 - self.exploration_noise_decrement) if self.noise > self.noise_min else self.noise_min