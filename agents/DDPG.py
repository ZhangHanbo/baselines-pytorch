import torch
from torch.autograd import Variable
from torch import optim
from torch import nn
import torch.nn.functional as F
import numpy as np
from agents.Agent import Agent
from config import DDPG_CONFIG
import basenets
import copy
from utils import databuffer

class DDPG(Agent):
    def __init__(self,hyperparams):
        config = copy.deepcopy(DDPG_CONFIG)
        config.update(hyperparams)
        super(DDPG, self).__init__(config)
        self.action_bounds = config['action_bounds']
        self.noise = config['noise_var']
        self.batch_size = config['batch_size']
        self.exploration_noise_decrement = config['noise_decrease']
        self.noise_min = config['noise_min']
        self.lra = config['lr_a']
        self.replace_tau = config['tau']
        # initialize zero memory [s, a, r, s_]
        config['memory_size'] = self.memory_size
        self.memory = databuffer(config)
        self.e_Actor = basenets.MLP(self.n_states, self.n_action_dims,
                                    n_hiddens=config['hidden_layers'],
                                    usebn=config['use_batch_norm'],
                                    outactive=F.tanh,
                                    outscaler=self.action_bounds)
        self.t_Actor = basenets.MLP(self.n_states, self.n_action_dims,
                                    n_hiddens=config['hidden_layers'],
                                    usebn=config['use_batch_norm'],
                                    outactive=F.tanh,
                                    outscaler=self.action_bounds)
        self.hard_update(self.t_Actor, self.e_Actor)
        self.e_Critic = basenets.MLP(self.n_states + self.n_action_dims, 1,
                                     n_hiddens=config['hidden_layers'],
                                     usebn=config['use_batch_norm'])
        self.t_Critic = basenets.MLP(self.n_states + self.n_action_dims, 1,
                                     n_hiddens=config['hidden_layers'],
                                     usebn=config['use_batch_norm'])
        self.hard_update(self.t_Critic, self.e_Critic)
        self.loss_func = config['critic_loss']()
        self.optimizer_a = config['optimizer_a'](self.e_Actor.parameters(), lr = self.lra)
        self.optimizer_c = config['optimizer_c'](self.e_Critic.parameters(), lr = self.lr)

    def choose_action(self,s):
        self.e_Actor.eval()
        s = torch.Tensor(s)
        anoise = torch.normal(torch.zeros(self.n_action_dims), self.noise * torch.ones(self.n_action_dims))
        preda = self.e_Actor(s)
        self.e_Actor.train()
        return np.array((preda + anoise).detach()).squeeze()

    def learn(self):
        # check to replace target parameters
        self.soft_update(self.t_Actor, self.e_Actor, self.replace_tau)
        self.soft_update(self.t_Critic, self.e_Critic, self.replace_tau)

        # sample batch memory from all memory
        batch_memory = self.sample_batch(self.batch_size)[0]
        r = torch.Tensor(batch_memory['reward'])
        done = torch.Tensor(batch_memory['done'])
        s_ = torch.Tensor(batch_memory['next_state'])
        a = torch.Tensor(batch_memory['action'])
        s = torch.Tensor(batch_memory['state'])

        q_target = r + self.gamma * self.t_Critic(torch.cat([s_, self.t_Actor(s_)], 1))
        q_target = q_target.detach()
        q_eval = self.e_Critic(torch.cat([s, a],1))

        # update critic
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

