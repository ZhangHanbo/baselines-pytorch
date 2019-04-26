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
import os

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
        self.lrv = config['lr_v']
        self.replace_tau = config['tau']
        # initialize zero memory [s, a, r, s_]
        config['memory_size'] = self.memory_size
        self.memory = databuffer(config)
        self.e_Actor = basenets.MLP(self.n_states, self.n_action_dims,
                                    n_hiddens=config['hidden_layers'],
                                    usebn=config['use_batch_norm'],
                                    nonlinear=config['act_func'],
                                    outactive=config['out_act_func'],
                                    outscaler=self.action_bounds)
        self.t_Actor = basenets.MLP(self.n_states, self.n_action_dims,
                                    n_hiddens=config['hidden_layers'],
                                    usebn=config['use_batch_norm'],
                                    nonlinear=config['act_func'],
                                    outactive=config['out_act_func'],
                                    outscaler=self.action_bounds)
        self.hard_update(self.t_Actor, self.e_Actor)
        self.e_Critic = basenets.MLP(self.n_states + self.n_action_dims, 1,
                                     n_hiddens=config['hidden_layers_v']
                                             if isinstance(config['hidden_layers_v'], list)
                                             else config['hidden_layers'],
                                     usebn=config['use_batch_norm'])
        self.t_Critic = basenets.MLP(self.n_states + self.n_action_dims, 1,
                                     n_hiddens=config['hidden_layers_v']
                                             if isinstance(config['hidden_layers_v'], list)
                                             else config['hidden_layers'],
                                     usebn=config['use_batch_norm'])
        self.hard_update(self.t_Critic, self.e_Critic)
        self.loss_func = config['critic_loss']()
        self.optimizer_a = config['optimizer_a'](self.e_Actor.parameters(), lr = self.lr)
        self.optimizer_c = config['optimizer_c'](self.e_Critic.parameters(), lr = self.lrv)

    def cuda(self):
        Agent.cuda(self)
        self.e_Actor = self.e_Actor.cuda()
        self.e_Critic = self.e_Critic.cuda()
        self.t_Actor = self.t_Actor.cuda()
        self.t_Critic = self.t_Critic.cuda()

    def choose_action(self, s):
        self.e_Actor.eval()
        s = torch.Tensor(s)
        anoise = torch.normal(torch.zeros(self.n_action_dims), self.noise * torch.ones(self.n_action_dims))
        if self.use_cuda:
            s = s.cuda()
            anoise = anoise.cuda()
        preda = self.e_Actor(s).detach()
        self.e_Actor.train()
        return (preda + anoise).cpu().squeeze().numpy()

    def learn(self):
        # check to replace target parameters
        self.soft_update(self.t_Actor, self.e_Actor, self.replace_tau)
        self.soft_update(self.t_Critic, self.e_Critic, self.replace_tau)

        # sample batch memory from all memory
        batch_memory = self.sample_batch(self.batch_size)[0]
        self.r = self.r.resize_(batch_memory['reward'].shape).copy_(torch.Tensor(batch_memory['reward']))
        self.done = self.done.resize_(batch_memory['done'].shape).copy_(torch.Tensor(batch_memory['done']))
        self.s_ = self.s_.resize_(batch_memory['next_state'].shape).copy_(torch.Tensor(batch_memory['next_state']))
        self.a = self.a.resize_(batch_memory['action'].shape).copy_(torch.Tensor(batch_memory['action']))
        self.s = self.s.resize_(batch_memory['state'].shape).copy_(torch.Tensor(batch_memory['state']))

        q_target = self.r + self.gamma * self.t_Critic(torch.cat([self.s_, self.t_Actor(self.s_)], 1))
        q_target = q_target.detach()
        q_eval = self.e_Critic(torch.cat([self.s, self.a],1))

        # update critic
        self.e_Critic.zero_grad()
        self.loss_c = self.loss_func(q_eval, q_target)
        self.loss_c.backward()
        self.optimizer_c.step()

        # update actor
        self.e_Actor.zero_grad()
        self.loss_a = -self.e_Critic(torch.cat([self.s, self.e_Actor(self.s)],1)).mean()
        self.loss_a.backward()
        self.optimizer_a.step()

        self.learn_step_counter += 1
        self.noise = self.noise * (
                1 - self.exploration_noise_decrement) if self.noise > self.noise_min else self.noise_min

    def save_model(self, save_path):
        print("saving models...")
        save_dict_a = {
            'model': self.e_Actor.state_dict(),
            'optimizer': self.optimizer_a.state_dict(),
            'noise': self.noise,
            'episode': self.episode_counter,
            'step': self.learn_step_counter,
        }
        save_dict_c = {
            'model': self.e_Critic.state_dict(),
            'optimizer': self.optimizer_c.state_dict(),
        }
        torch.save(save_dict_a, os.path.join(save_path, "actor" + str(self.learn_step_counter) + ".pth"))
        torch.save(save_dict_c, os.path.join(save_path, "critic" + str(self.learn_step_counter) + ".pth"))

    def load_model(self, load_path, load_point):
        actor_name = os.path.join(load_path, "actor" + str(load_point) + ".pth")
        print("loading checkpoint %s" % (actor_name))
        checkpoint = torch.load(actor_name)
        self.e_Actor.load_state_dict(checkpoint['model'])
        self.optimizer_a.load_state_dict(checkpoint['optimizer'])
        self.noise = checkpoint['noise']
        self.learn_step_counter = checkpoint['step']
        self.episode_counter = checkpoint['episode']
        print("loaded checkpoint %s" % (actor_name))

        critic_name = os.path.join(load_path, "critic" + str(load_point) + ".pth")
        print("loading checkpoint %s" % (critic_name))
        checkpoint = torch.load(critic_name)
        self.e_Critic.load_state_dict(checkpoint['model'])
        self.optimizer_c.load_state_dict(checkpoint['optimizer'])
        print("loaded checkpoint %s" % (critic_name))
