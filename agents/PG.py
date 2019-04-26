import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
import basenets
from agents.Agent import Agent
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
import copy
from config import PG_CONFIG
from rlnets.PG import FCPG_Gaussian, FCPG_Softmax, FCVALUE
from utils import databuffer, databuffer_PG_gaussian, databuffer_PG_softmax
import abc
import os

class PG(Agent):
    __metaclass__ = abc.ABCMeta
    def __init__(self,hyperparams):
        config = copy.deepcopy(PG_CONFIG)
        config.update(hyperparams)
        super(PG, self).__init__(config)
        self.value_type = config['value_type']
        self.using_batch = config['using_batch']
        self.entropy_weight = config['entropy_weight']
        if self.value_type is not None:
            # initialize value network architecture
            if self.value_type == 'FC':
                self.value = FCVALUE(self.n_states,
                                     n_hiddens=config['hidden_layers_v']
                                                if isinstance(config['hidden_layers_v'], list)
                                                else config['hidden_layers'],
                                     usebn=config['use_batch_norm']
                                     )
            # choose estimator, including Q, A and GAE.
            self.lamb = config['GAE_lambda']
            # value approximator optimizer
            self.loss_func_v = config['loss_func_v']()
            self.lr_v = config['lr_v']
            self.mom_v = config['mom_v']
            if config['v_optimizer'] == optim.LBFGS:
                self.using_lbfgs_for_V = True
            else:
                self.using_lbfgs_for_V = False
                if self.mom is not None:
                    self.v_optimizer = config['v_optimizer'](self.value.parameters(), lr=self.lr_v, momentum = self.mom_v)
                else:
                    self.v_optimizer = config['v_optimizer'](self.value.parameters(), lr=self.lr_v)
        elif self.value_type is None:
            self.value = None
        # damping initialization, all the following will be used in PG algorithm
        self.policy = None
        self.optimizer = None
        self.A = None
        self.V = None
        self.sample_index = None
        if self.using_batch:
            self.s_batch = torch.FloatTensor(1)
            self.a_batch = torch.FloatTensor(1)
            self.r_batch = torch.FloatTensor(1)
            self.done_batch = torch.FloatTensor(1)

    def cuda(self):
        Agent.cuda(self)
        self.policy = self.policy.cuda()
        if self.value_type is not None:
            self.value = self.value.cuda()
        if self.using_batch:
            self.s_batch = self.s_batch.cuda()
            self.a_batch = self.a_batch.cuda()
            self.r_batch = self.r_batch.cuda()
            self.done_batch = self.done_batch.cuda()

    @abc.abstractmethod
    def sample_batch(self, batchsize = None):
        raise NotImplementedError("Must be implemented in subclass.")

    # compute importance sampling factor between current policy and previous trajectories
    @abc.abstractmethod
    def compute_imp_fac(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def compute_entropy(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def estimate_value(self):
        masks = 1 - self.done
        returns = torch.zeros(self.r.size(0), 1).type_as(self.r)
        prev_return = 0
        # using value approximator
        if self.value_type is not None:
            values = self.value(self.s)
            prev_value = 0
            delta = torch.zeros(self.r.size(0), 1).type_as(self.r)
            advantages = torch.zeros(self.r.size(0), 1).type_as(self.r)
            prev_advantage = 0
            for i in reversed(range(self.r.size(0))):
                delta[i] = self.r[i] + self.gamma * masks[i] * prev_value - values[i]
                advantages[i] = delta[i] + self.gamma * self.lamb * masks[i] * prev_advantage
                returns[i] = self.r[i] + self.gamma * prev_return * masks[i]
                prev_value = float(values[i, 0])
                prev_advantage = float(advantages[i, 0])
                prev_return = float(returns[i, 0])
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # values and advantages are all 2-D Tensor. size: r.size(0) x 1
            self.V = returns.squeeze()
            self.V = self.V.detach()
            self.A = advantages.squeeze()
        else:
            for i in reversed(range(self.r.size(0))):
                returns[i] = self.r[i] + self.gamma * prev_return * masks[i]
                prev_return = float(returns[i, 0])
            self.V = returns.squeeze()
            self.V = (self.V - self.V.mean()) / (self.V.std() + 1e-8)

    def optim_value_lbfgs(self,V_target):
        value = self.value
        value.zero_grad()
        loss_fn = self.loss_func_v
        def V_closure():
            if self.using_batch:
                predicted = value(self.s_batch).squeeze()
            else:
                predicted = value(self.s).squeeze()
            loss = loss_fn(predicted, V_target)
            optimizer.zero_grad()
            loss.backward()
            return loss
        old_params = parameters_to_vector(value.parameters())
        for lr in self.lr * .5 ** np.arange(10):
            optimizer = optim.LBFGS(self.value.parameters(), lr=lr)
            optimizer.step(V_closure)
            current_params = parameters_to_vector(value.parameters())
            if any(np.isnan(current_params.data.cpu().numpy())):
                print("LBFGS optimization diverged. Rolling back update...")
                vector_to_parameters(old_params, value.parameters())
            else:
                return

    def update_value(self):
        V_target = self.V
        if self.using_batch:
            V_target = Variable(V_target.data[self.sample_index])
        if self.using_lbfgs_for_V:
            self.optim_value_lbfgs(V_target)
        else:
            V_eval = self.value(self.s_batch if self.using_batch else self.s)
            self.loss_v = self.loss_func_v(V_eval.squeeze(), V_target)
            self.value.zero_grad()
            self.loss_v.backward()
            self.v_optimizer.step()

    def learn(self):
        self.sample_batch()
        self.estimate_value()
        imp_fac = self.compute_imp_fac()
        # update policy
        if self.value_type is not None:
            self.loss = - (imp_fac * self.A.squeeze()).mean()
            # update value
            self.update_value()
        else:
            self.loss = - (imp_fac * self.V.squeeze()).mean()
        self.policy.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1

    def save_model(self, save_path):
        print("saving models...")
        save_dict = {
            'model': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode': self.episode_counter,
            'step': self.learn_step_counter,
        }
        torch.save(save_dict, os.path.join(save_path, "policy" + str(self.learn_step_counter) + ".pth"))
        if self.value_type is not None and not self.using_lbfgs_for_V:
            save_dict = {
                'model': self.value.state_dict(),
                'optimizer': self.v_optimizer.state_dict(),
            }
            torch.save(save_dict, os.path.join(save_path, "value" + str(self.learn_step_counter) + ".pth"))

    def load_model(self, load_path, load_point):
        policy_name = os.path.join(load_path, "policy" + str(load_point) + ".pth")
        print("loading checkpoint %s" % (policy_name))
        checkpoint = torch.load(policy_name)
        self.policy.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.learn_step_counter = checkpoint['step']
        self.episode_counter = checkpoint['episode']
        print("loaded checkpoint %s" % (policy_name))

        if self.value_type is not None and not self.using_lbfgs_for_V:
            value_name = os.path.join(load_path, "value" + str(load_point) + ".pth")
            print("loading checkpoint %s" % (value_name))
            checkpoint = torch.load(value_name)
            self.value.load_state_dict(checkpoint['model'])
            self.v_optimizer.load_state_dict(checkpoint['optimizer'])
            print("loaded checkpoint %s" % (value_name))

class PG_Gaussian(PG):
    def __init__(self,hyperparams):
        config = copy.deepcopy(PG_CONFIG)
        config.update(hyperparams)
        super(PG_Gaussian, self).__init__(config)
        config['memory_size'] = self.memory_size
        self.memory = databuffer_PG_gaussian(config)
        self.action_bounds = config['action_bounds']
        self.policy = FCPG_Gaussian(self.n_states,  # input dim
                                   self.n_action_dims,  # output dim
                                   sigma=config['init_noise'],
                                   outactive=config['out_act_func'],
                                   outscaler=self.action_bounds,
                                   n_hiddens=config['hidden_layers'],
                                   nonlinear=config['act_func'],
                                   usebn=config['use_batch_norm'])
        if self.mom is not None:
            self.optimizer = config['optimizer'](self.policy.parameters(), lr=self.lr, momontum = self.mom)
        else:
            self.optimizer = config['optimizer'](self.policy.parameters(), lr=self.lr)

        self.mu = torch.FloatTensor(1)
        self.sigma = torch.FloatTensor(1)
        if self.using_batch:
            self.mu_batch = torch.FloatTensor(1)
            self.sigma_batch = torch.FloatTensor(1)

    def cuda(self):
        PG.cuda(self)
        self.mu = self.mu.cuda()
        self.sigma = self.sigma.cuda()
        if self.using_batch:
            self.mu_batch = self.mu_batch.cuda()
            self.sigma_batch = self.sigma_batch.cuda()

    def choose_action(self, s):
        self.policy.eval()
        s = Variable(torch.Tensor(s))
        if self.use_cuda:
            s = s.cuda()
        mu,sigma = self.policy(s)
        mu = mu.detach()
        sigma = sigma.detach()
        self.policy.train()
        a = torch.normal(mu,sigma)
        return a.cpu().numpy(), mu.cpu().numpy(), sigma.cpu().numpy()

    def compute_logp(self,mu,sigma,a):
        if a.dim() == 1:
            return torch.sum(torch.pow((a - mu),2.) / (- 2. * sigma)) - \
                    self.n_action_dims / 2. * torch.log(torch.FloatTensor([2. * 3.14159]).type_as(mu)) - \
                    1. / 2. * torch.sum(torch.log(sigma))
        elif a.dim() == 2:
            return  torch.sum(torch.pow((a - mu), 2.) / (- 2. * sigma),1) - \
                    self.n_action_dims / 2. * torch.log(torch.FloatTensor([2. * 3.14159]).type_as(mu)) - \
                    1. / 2. * torch.sum(torch.log(sigma),1)
        else:
            RuntimeError("a must be a 1-D or 2-D Tensor or Variable")

    def compute_imp_fac(self):
        # theta is the vectorized model parameters
        if self.using_batch:
            mucur, sigmacur = self.policy(self.s_batch)
            # important sampling coefficients
            # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
            imp_fac = torch.exp(
                self.compute_logp(mucur, sigmacur, self.a_batch) - self.compute_logp(self.mu_batch, self.sigma_batch, self.a_batch))
        else:
            mucur, sigmacur = self.policy(self.s)
            # important sampling coefficients
            # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
            imp_fac = torch.exp(
                self.compute_logp(mucur, sigmacur, self.a) - self.compute_logp(self.mu, self.sigma, self.a))
        return imp_fac

    def compute_entropy(self):
        if self.using_batch:
            mucur, sigmacur = self.policy(self.s_batch)
            # important sampling coefficients
            # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
            entropy = 1./2. * (self.n_action_dims * 2.838 + torch.sum(torch.log(sigmacur), 1)).mean()
        else:
            mucur, sigmacur = self.policy(self.s)
            # important sampling coefficients
            # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
            entropy = 1./2. * (self.n_action_dims * 2.838 + torch.sum(torch.log(sigmacur), 1)).mean()
        return entropy

    def sample_batch(self, batch_size = None):
        if batch_size is not None:
            batch, self.sample_index = Agent.sample_batch(self, batch_size)
            self.r_batch = self.r_batch.resize_(batch['reward'].shape).copy_(torch.Tensor(batch['reward']))
            self.done_batch = self.done_batch.resize_(batch['done'].shape).copy_(torch.Tensor(batch['done']))
            self.a_batch = self.a_batch.resize_(batch['action'].shape).copy_(torch.Tensor(batch['action']))
            self.s_batch = self.s_batch.resize_(batch['state'].shape).copy_(torch.Tensor(batch['state']))
            self.mu_batch = self.mu_batch.resize_(batch['mu'].shape).copy_(torch.Tensor(batch['mu']))
            self.sigma_batch = self.sigma_batch.resize_(batch['sigma'].shape).copy_(torch.Tensor(batch['sigma']))
        else:
            batch, self.sample_index = Agent.sample_batch(self)
            self.r = self.r.resize_(batch['reward'].shape).copy_(torch.Tensor(batch['reward']))
            self.done = self.done.resize_(batch['done'].shape).copy_(torch.Tensor(batch['done']))
            self.a = self.a.resize_(batch['action'].shape).copy_(torch.Tensor(batch['action']))
            self.s = self.s.resize_(batch['state'].shape).copy_(torch.Tensor(batch['state']))
            self.mu = self.mu.resize_(batch['mu'].shape).copy_(torch.Tensor(batch['mu']))
            self.sigma = self.sigma.resize_(batch['sigma'].shape).copy_(torch.Tensor(batch['sigma']))

class PG_Softmax(PG):
    def __init__(self,hyperparams):
        config = copy.deepcopy(PG_CONFIG)
        config.update(hyperparams)
        super(PG_Softmax, self).__init__(config)
        # 3 is a, r, done, n_actions is the distribution.
        config['memory_size'] = self.memory_size
        self.memory = databuffer_PG_softmax(config)
        self.policy = FCPG_Softmax(self.n_states,  # input dim
                                   self.n_actions,  # output dim
                                   n_hiddens=config['hidden_layers'],
                                   nonlinear=config['act_func'],
                                   outactive=config['out_act_func'],
                                   usebn=config['use_batch_norm'],
                                   )
        if self.mom is not None:
            self.optimizer = config['optimizer'](self.policy.parameters(), lr=self.lr, momontum = self.mom)
        else:
            self.optimizer = config['optimizer'](self.policy.parameters(), lr=self.lr)

        self.distri = torch.FloatTensor(1)
        if self.using_batch:
            self.distri_batch = torch.FloatTensor(1)

    def cuda(self):
        PG.cuda(self)
        self.distri = self.distri.cuda()
        if self.using_batch:
            self.distri_batch = self.distri_batch.cuda()

    def choose_action(self, s):
        self.policy.eval()
        s = Variable(torch.Tensor(s))
        if self.use_cuda:
            s = s.cuda()
        distri = self.policy(s).detach()
        self.policy.train()
        a = np.random.choice(distri.shape[0], p = distri.cpu().numpy())
        return a, distri.cpu().numpy()

    def compute_logp(self,distri,a):
        if distri.dim() == 1:
            return torch.log(distri[a])
        elif distri.dim() == 2:
            a_indices = [range(distri.size(0)),list(a.data.squeeze().long())]
            return torch.log(distri[a_indices])
        else:
            RuntimeError("distri must be a 1-D or 2-D Tensor or Variable")

    def compute_imp_fac(self):
        if self.using_batch:
            distri = self.policy(self.s_batch)
            # important sampling coefficients
            # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
            imp_fac = torch.exp(self.compute_logp(distri, self.a_batch) - self.compute_logp(self.distri_batch, self.a_batch))
        else:
            distri = self.policy(self.s)
            # important sampling coefficients
            # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
            imp_fac = torch.exp(self.compute_logp(distri, self.a) - self.compute_logp(self.distri, self.a))
        return imp_fac

    def compute_entropy(self):
        if self.using_batch:
            distri = self.policy(self.s_batch)
            # important sampling coefficients
            # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
            entropy = - torch.sum(distri * torch.log(distri),1).mean()
        else:
            distri = self.policy(self.s)
            # important sampling coefficients
            # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
            entropy = - torch.sum(distri * torch.log(distri), 1).mean()
        return entropy

    def sample_batch(self, batch_size = None):
        if batch_size is not None:
            batch, self.sample_index = Agent.sample_batch(self, batch_size)
            self.r_batch = self.r_batch.resize_(batch['reward'].shape).copy_(torch.Tensor(batch['reward']))
            self.done_batch = self.done_batch.resize_(batch['done'].shape).copy_(torch.Tensor(batch['done']))
            self.a_batch = self.a_batch.resize_(batch['action'].shape).copy_(torch.Tensor(batch['action']))
            self.s_batch = self.s_batch.resize_(batch['state'].shape).copy_(torch.Tensor(batch['state']))
            self.distri_batch = self.distri_batch.resize_(batch['distr'].shape).copy_(torch.Tensor(batch['distr']))
        else:
            batch, self.sample_index = Agent.sample_batch(self)
            self.r = self.r.resize_(batch['reward'].shape).copy_(torch.Tensor(batch['reward']))
            self.done = self.done.resize_(batch['done'].shape).copy_(torch.Tensor(batch['done']))
            self.a = self.a.resize_(batch['action'].shape).copy_(torch.Tensor(batch['action']))
            self.s = self.s.resize_(batch['state'].shape).copy_(torch.Tensor(batch['state']))
            self.distri = self.distri.resize_(batch['distr'].shape).copy_(torch.Tensor(batch['distr']))
