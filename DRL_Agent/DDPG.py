import torch
from torch.autograd import Variable
from torch import optim
from torch import nn
import torch.nn.functional as F
import numpy as np
from DRL_Agent.Agent import Agent
from config import DDPG_CONFIG
import Feature_Extractor
import copy

class DDPG(Agent):
    def __init__(self,hyperparams):
        config = copy.deepcopy(DDPG_CONFIG)
        config.update(hyperparams)
        super(DDPG, self).__init__(config)
        self.action_bounds = config['action_bounds']
        self.noise = config['noise_var']
        self.exploration_noise_decrement = config['noise_decrease']
        self.noise_min = config['noise_min']
        self.lra = config['lr_a']
        self.replace_tau = config['tau']
        # initialize zero memory [s, a, r, s_]
        self.memory = torch.Tensor(np.zeros((self.memory_size, self.n_features * 2 + 2 + self.n_actions)))
        self.e_Actor = Feature_Extractor.MLP(self.n_features, self.n_actions, outactive=F.tanh, outscaler=self.action_bounds)
        self.t_Actor = Feature_Extractor.MLP(self.n_features, self.n_actions, outactive=F.tanh, outscaler=self.action_bounds)
        self.e_Critic = Feature_Extractor.MLP(self.n_features + self.n_actions, 1)
        self.t_Critic = Feature_Extractor.MLP(self.n_features + self.n_actions, 1)
        self.loss_func = config['critic_loss']()
        self.optimizer_a = config['optimizer_a'](self.e_Actor.parameters(), lr = self.lra)
        self.optimizer_c = config['optimizer_c'](self.e_Critic.parameters(), lr = self.lr)

    def choose_action(self,s):
        s = Variable(torch.Tensor(s))
        anoise = torch.normal(torch.zeros(self.n_actions),self.noise * torch.ones(self.n_actions))
        preda = self.e_Actor(s).data
        return np.array(preda + anoise)

    def learn(self):
        # check to replace target parameters
        self.soft_update(self.t_Actor, self.e_Actor, self.replace_tau)
        self.soft_update(self.t_Critic, self.e_Critic, self.replace_tau)

        # sample batch memory from all memory
        batch_memory = self.sample_batch()

        # update critic
        r = Variable(batch_memory[:, (self.n_features + self.n_actions):(self.n_features + self.n_actions+ 1)])

        done = Variable(batch_memory[:, (self.n_features + self.n_actions +1):(self.n_features + self.n_actions+ 2)])
        s_ = Variable(batch_memory[:, -self.n_features:])
        q_target = r + self.gamma * self.t_Critic(torch.cat([s_, self.t_Actor(s_)],1))
        q_target = q_target.detach()

        a = Variable(batch_memory[:, self.n_features:(self.n_actions +self.n_features)])
        s = Variable(batch_memory[:, :self.n_features])
        q_eval = self.e_Critic(torch.cat([s, a],1))

        self.e_Critic.zero_grad()
        self.loss_c = self.loss_func(q_eval, q_target)
        self.loss_c.backward()
        self.optimizer_c.step()

        # update actor
        self.e_Actor.zero_grad()
        self.loss_a = -self.e_Critic(torch.cat([s, self.e_Actor(s)],1)).mean()
        self.loss_a.backward()
        self.optimizer_a.step()

        self.learn_step_counter += 1
        self.noise = self.noise * (
                1 - self.exploration_noise_decrement) if self.noise > self.noise_min else self.noise_min

