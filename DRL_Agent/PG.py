import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
import Feature_Extractor
from DRL_Agent.Agent import Agent
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
import copy
from config import PG_CONFIG
from Appro_Func.PG import FCPG_Gaussian, FCPG_Softmax, FCVALUE
import abc

class PG(Agent):
    __metaclass__ = abc.ABCMeta
    def __init__(self,hyperparams):
        config = copy.deepcopy(PG_CONFIG)
        config.update(hyperparams)
        super(PG, self).__init__(config)
        self.value_type = config['value_type']
        if self.value_type is not None:
            # initialize value network architecture
            if self.value_type == 'FC':
                self.value = FCVALUE(self.n_features)
            # choose estimator, including Q, A and GAE.
            self.lamb = config['GAE_lambda']
            # value approximator optimizer
            self.loss_func_v = config['loss_func_v']()
            if config['v_optimizer'] == optim.LBFGS:
                self.using_lbfgs_for_V = True
            else:
                self.using_lbfgs_for_V = False
                self.v_optimizer = config['v_optimizer'](self.value.parameters(), lr=self.lr, momentum = self.mom)
        elif self.value_type is None:
            self.value = None
        # damping initialization, all the following will be used in PG algorithm
        self.s = Variable(torch.Tensor([]))
        self.a = Variable(torch.Tensor([]))
        self.r = Variable(torch.Tensor([]))
        self.done = Variable(torch.Tensor([]))
        self.policy = None
        self.optimizer = None
        self.A = None
        self.V = None
        self.sample_index = None

    @abc.abstractmethod
    def sample_batch(self, batchsize = None):
        raise NotImplementedError("Must be implemented in subclass.")

    # compute importance sampling factor between current policy and previous trajectories
    @abc.abstractmethod
    def compute_imp_fac(self, using_batch = False):
        raise NotImplementedError("Must be implemented in subclass.")

    def estimate_value(self):
        masks = 1 - self.done
        returns = Variable(torch.zeros(self.r.size(0), 1))
        prev_return = 0
        # using value approximator
        if self.value_type is not None:
            values = self.value(self.s)
            prev_value = 0
            delta = Variable(torch.zeros(self.r.size(0), 1))
            advantages = Variable(torch.zeros(self.r.size(0), 1))
            prev_advantage = 0
            for i in reversed(range(self.r.size(0))):
                returns[i] = self.r[i] + self.gamma * prev_return * masks[i]
                delta[i] = self.r[i] + self.gamma * masks[i] * prev_value - values[i]
                advantages[i] = delta[i] + self.gamma * self.lamb * masks[i] * prev_advantage
                prev_return = float(returns[i, 0])
                prev_value = float(values[i, 0])
                prev_advantage = float(advantages[i, 0])
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # values and advantages are all 2-D Tensor. size: r.size(0) x 1
            self.V = returns.squeeze()
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
            predicted = value(self.s)
            loss = loss_fn(predicted, V_target)
            optimizer.zero_grad()
            loss.backward()
            return loss
        old_params = parameters_to_vector(value.parameters())
        for lr in self.lr * .5 ** np.arange(10):
            optimizer = optim.LBFGS(self.value.parameters(), lr = lr)
            optimizer.step(V_closure)
            current_params = parameters_to_vector(value.parameters())
            if any(np.isnan(current_params.data.cpu().numpy())):
                print("LBFGS optimization diverged. Rolling back update...")
                vector_to_parameters(old_params, value.parameters())
            else:
                return

    def update_value(self):
        V_target = self.V
        if self.using_lbfgs_for_V:
            self.optim_value_lbfgs(V_target)
        else:
            s = Variable(self.memory[:, :self.n_features])
            V_eval = self.value(s)
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

class PG_Gaussian(PG):
    def __init__(self,hyperparams):
        config = copy.deepcopy(PG_CONFIG)
        config.update(hyperparams)
        super(PG_Gaussian, self).__init__(config)
        self.memory = torch.Tensor(np.zeros((self.memory_size, self.n_features * 2 + 2 + 3 * self.n_actions)))
        self.action_bounds = config['action_bounds']
        self.policy = FCPG_Gaussian(self.n_features,  # input dim
                                   self.n_actions,  # output dim
                                   sigma=2,
                                   outactive=F.tanh,
                                   outscaler=self.action_bounds
                                   )
        if self.mom is not None:
            self.optimizer = config['optimizer'](self.policy.parameters(), lr=self.lr, momontum = self.mom)
        else:
            self.optimizer = config['optimizer'](self.policy.parameters(), lr=self.lr)

    def choose_action(self, s):
        s = Variable(torch.Tensor(s))
        mu,sigma = self.policy(s)
        a = torch.normal(mu,sigma)
        return np.array(a.data), np.array(mu.data), np.array(sigma.data)

    def compute_logp(self,mu,sigma,a):
        if a.dim() == 1:
            return torch.sum(torch.pow((a - mu),2) / (- 2 * sigma)) - \
                    self.n_actions / 2 * torch.log(Variable(torch.Tensor([2*3.14159]))) - \
                    1 / 2 * torch.sum(torch.log(sigma))
        elif a.dim() == 2:
            return torch.sum(torch.pow((a - mu), 2) / (- 2 * sigma),1) - \
                    self.n_actions / 2 * torch.log(Variable(torch.Tensor([2 * 3.14159]))) - \
                    1 / 2 * torch.sum(torch.log(sigma),1)
        else:
            RuntimeError("a must be a 1-D or 2-D Tensor or Variable")

    def compute_imp_fac(self, using_batch = False):
        # theta is the vectorized model parameters
        if using_batch:
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

    def sample_batch(self, batch_size = None):
        if batch_size is not None:
            batch, self.sample_index = Agent.sample_batch(self)
            self.s_batch = Variable(batch[:, :self.n_features])
            self.a_batch = Variable(
                batch[:, 2 * self.n_actions + self.n_features: 3 * self.n_actions + self.n_features])
            self.mu_batch = Variable(batch[:, self.n_features:(self.n_actions + self.n_features)])
            self.sigma_batch = Variable(
                batch[:, self.n_actions + self.n_features: 2 * self.n_actions + self.n_features])
            self.r_batch = Variable(
                batch[:, (self.n_features + 3 * self.n_actions):(self.n_features + 3 * self.n_actions + 1)])
            self.done_batch = Variable(
                batch[:, (self.n_features + 3 * self.n_actions + 1):(self.n_features + 3 * self.n_actions + 2)])
        else:
            if self.memory_counter > self.memory_size:
                self.s = Variable(self.memory[:, :self.n_features])
                self.a = Variable(
                    self.memory[:, 2 * self.n_actions + self.n_features: 3 * self.n_actions + self.n_features])
                self.mu = Variable(self.memory[:, self.n_features:(self.n_actions + self.n_features)])
                self.sigma = Variable(
                    self.memory[:, self.n_actions + self.n_features: 2 * self.n_actions + self.n_features])
                self.r = Variable(
                    self.memory[:, (self.n_features + 3 * self.n_actions):(self.n_features + 3 * self.n_actions + 1)])
                self.done = Variable(
                    self.memory[:,
                    (self.n_features + 3 * self.n_actions + 1):(self.n_features + 3 * self.n_actions + 2)])
            else:
                self.s = Variable(self.memory[:self.memory_counter, :self.n_features])
                self.a = Variable(
                    self.memory[:self.memory_counter,
                    2 * self.n_actions + self.n_features: 3 * self.n_actions + self.n_features])
                self.mu = Variable(
                    self.memory[:self.memory_counter, self.n_features:(self.n_actions + self.n_features)])
                self.sigma = Variable(
                    self.memory[:self.memory_counter,
                    self.n_actions + self.n_features: 2 * self.n_actions + self.n_features])
                self.r = Variable(
                    self.memory[:self.memory_counter,
                    (self.n_features + 3 * self.n_actions):(self.n_features + 3 * self.n_actions + 1)])
                self.done = Variable(
                    self.memory[:self.memory_counter,
                    (self.n_features + 3 * self.n_actions + 1):(self.n_features + 3 * self.n_actions + 2)])

class PG_Softmax(PG):
    def __init__(self,hyperparams):
        config = copy.deepcopy(PG_CONFIG)
        config.update(hyperparams)
        super(PG_Softmax, self).__init__(config)
        # 3 is a, r, done, n_actions is the distribution.
        self.memory = torch.Tensor(np.zeros((self.memory_size, self.n_features * 2 + 3 + self.n_actions)))
        self.policy = FCPG_Softmax(self.n_features,  # input dim
                                   self.n_actions,  # output dim
                                   )
        if self.mom is not None:
            self.optimizer = config['optimizer'](self.policy.parameters(), lr=self.lr, momontum = self.mom)
        else:
            self.optimizer = config['optimizer'](self.policy.parameters(), lr=self.lr)

    def choose_action(self, s):
        s = Variable(torch.Tensor(s))
        distri = self.policy(s)
        a = np.random.choice(distri.data.shape[0], p = np.array(distri.data))
        return a, np.array(distri.data)

    def compute_logp(self,distri,a):
        if distri.dim() == 1:
            return torch.log(distri[a])
        elif distri.dim() == 2:
            a_indices = [range(distri.size(0)),list(a.data.squeeze().long())]
            return torch.log(distri[a_indices])
        else:
            RuntimeError("distri must be a 1-D or 2-D Tensor or Variable")

    def compute_imp_fac(self, using_batch = False):
        if using_batch:
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

    def sample_batch(self, batch_size = None):
        if batch_size is not None:
            batch, self.sample_index = Agent.sample_batch(self)
            self.s_batch = Variable(batch[:, :self.n_features])
            self.a_batch = Variable(batch[:, self.n_actions + self.n_features: self.n_actions + self.n_features + 1])
            self.distri_batch = Variable(batch[:, self.n_features:self.n_actions + self.n_features])
            self.r_batch = Variable(
                batch[:, (self.n_actions + self.n_features + 1):(self.n_actions + self.n_features + 2)])
            self.done_batch = Variable(
                batch[:, (self.n_actions + self.n_features + 2):(self.n_actions + self.n_features + 3)])
        else:
            if self.memory_counter > self.memory_size:
                self.s = Variable(self.memory[:, :self.n_features])
                self.a = Variable(
                    self.memory[:, self.n_actions + self.n_features: self.n_actions + self.n_features + 1])
                self.distri = Variable(self.memory[:, self.n_features:self.n_actions + self.n_features])
                self.r = Variable(
                    self.memory[:, (self.n_actions + self.n_features + 1):(self.n_actions + self.n_features + 2)])
                self.done = Variable(
                    self.memory[:, (self.n_actions + self.n_features + 2):(self.n_actions + self.n_features + 3)])
            else:
                self.s = Variable(self.memory[:self.memory_counter, :self.n_features])
                self.a = Variable(self.memory[:self.memory_counter,
                                  self.n_actions + self.n_features: self.n_actions + self.n_features + 1])
                self.distri = Variable(
                    self.memory[:self.memory_counter, self.n_features:self.n_actions + self.n_features])
                self.r = Variable(
                    self.memory[:self.memory_counter,
                    (self.n_actions + self.n_features + 1):(self.n_actions + self.n_features + 2)])
                self.done = Variable(
                    self.memory[:self.memory_counter,
                    (self.n_actions + self.n_features + 2):(self.n_actions + self.n_features + 3)])