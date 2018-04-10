from PG import PG, PG_Softmax, PG_Gaussian
from NPG import NPG, NPG_Softmax, NPG_Gaussian
import torch
import torch.nn
from torch.autograd import Variable
import numpy as np
from Agent import Agent
from config import AdaptiveKLPPO_CONFIG,PPO_CONFIG
import copy
import abc

class PPO(NPG):
    __metaclass__ = abc.ABCMeta
    def __init__(self, hyperparams):
        config = copy.deepcopy(PPO_CONFIG)
        config.update(hyperparams)
        super(PPO, self).__init__(config)
        self.steps = config['steps_per_update']
        self.epsilon = config['clip_epsilon']

    def learn(self):
        self.sample_batch()
        self.estimate_value()
        # update value
        self.update_value()
        # update policy
        for istep in range(self.steps):
            self.sample_batch(self.batch_size)
            # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
            imp_fac = self.compute_imp_fac( using_batch = True)
            # values and advantages are all 2-D Tensor. size: r.size(0) x 1
            if self.A is not None:
                self.A_batch = Variable(self.A[self.sample_index].data)
                self.loss1 = imp_fac * self.A_batch
                self.loss2 = torch.clamp(imp_fac,1-self.epsilon, 1+self.epsilon) * self.A_batch + 1e-8
                self.loss = - torch.min(self.loss1, self.loss2).mean()
            else:
                self.V_batch = Variable(self.V[self.sample_index].data)
                self.loss1 = imp_fac * self.V_batch
                self.loss2 = torch.clamp(imp_fac, 1 - self.epsilon, 1 + self.epsilon) * self.V_batch
                self.loss = - torch.min(self.loss1, self.loss2).mean()
            self.policy.zero_grad()
            self.loss.backward()
            self.optimizer.step()
        self.learn_step_counter += 1

class PPO_Gaussian(PPO, NPG_Gaussian):
    def __init__(self,hyperparams):
        super(PPO_Gaussian,self).__init__(hyperparams)

class PPO_Softmax(PPO, NPG_Softmax):
    def __init__(self,hyperparams):
        super(PPO_Softmax,self).__init__(hyperparams)


# adaptive KL PPO is the same with NPG except that KL bound will adaptively increase or decrease after each update
class AdaptiveKLPPO(NPG):
    __metaclass__ = abc.ABCMeta
    def __init__(self,hyperparams):
        config = copy.deepcopy(AdaptiveKLPPO_CONFIG)
        config.update(hyperparams)
        super(AdaptiveKLPPO, self).__init__(config)
        self.beta = config['init_beta']
        self.steps = config['steps_per_update']
        self.cur_kl = 0

    def update_beta(self):
        cur_kl = self.cur_kl.data[0]
        if cur_kl < self.max_kl / 1.5:
            self.beta /= 2
        else:
            self.beta *= 2

    def learn(self):
        self.sample_batch()
        self.estimate_value()
        # update value
        self.update_value()
        # update policy
        for istep in range(self.steps):
            self.sample_batch(self.batch_size)
            # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
            imp_fac = self.compute_imp_fac( using_batch=True)
            # update policy
            cur_kl = self.mean_kl_divergence(using_batch = True)
            if self.A is not None:
                self.A_batch = Variable(self.A[self.sample_index].data)
                self.loss = - (imp_fac * self.A_batch).mean() + self.beta * cur_kl
            else:
                self.V_batch = Variable(self.V[self.sample_index].data)
                self.loss = - (imp_fac * self.V_batch).mean() + self.beta * cur_kl
            self.policy.zero_grad()
            self.loss.backward()
            self.optimizer.step()
        # update beta
        self.cur_kl = self.mean_kl_divergence()
        self.update_beta()
        self.learn_step_counter += 1

class AdaptiveKLPPO_Gaussian(AdaptiveKLPPO, NPG_Gaussian):
    def __init__(self,hyperparams):
        super(AdaptiveKLPPO_Gaussian,self).__init__(hyperparams)

class AdaptiveKLPPO_Softmax(AdaptiveKLPPO, NPG_Softmax):
    def __init__(self,hyperparams):
        super(AdaptiveKLPPO_Softmax,self).__init__(hyperparams)


