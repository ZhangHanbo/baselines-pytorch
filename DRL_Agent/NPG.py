from DRL_Agent.PG import PG, PG_Gaussian, PG_Softmax
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
import Feature_Extractor
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
import copy
from config import NPG_CONFIG
import abc

class NPG(PG):
    __metaclass__ = abc.ABCMeta
    def __init__(self,hyperparams):
        config = copy.deepcopy(NPG_CONFIG)
        config.update(hyperparams)
        super(NPG, self).__init__(config)
        self.cg_iters = config['cg_iters']
        self.cg_residual_tol = config['cg_residual_tol']
        self.cg_damping = config['cg_damping']
        self.max_kl = config['max_kl_divergence']

    def conjunction_gradient(self, b):
        """
        Demmel p 312, borrowed from https://github.com/ikostrikov/pytorch-trpo
        """
        p = b.clone().data
        r = b.clone().data
        x = torch.zeros_like(b).data
        rdotr = torch.sum(r * r)
        for i in xrange(self.cg_iters):
            z = self.hessian_vector_product(Variable(p))
            v = rdotr / torch.sum(p * z.data)
            x += v * p
            r -= v * z.data
            newrdotr = torch.sum(r * r)
            mu = newrdotr / rdotr
            p = r + mu * p
            rdotr = newrdotr
            if rdotr < self.cg_residual_tol:
                break
        return Variable(x)

    def hessian_vector_product(self, vector):
        """
        Returns the product of the Hessian of the KL divergence and the given vector
        Borrowed from https://github.com/mjacar/pytorch-trpo/blob/master/trpo_agent.py
        """
        self.policy.zero_grad()
        mean_kl_div = self.mean_kl_divergence()
        kl_grad = torch.autograd.grad(
            mean_kl_div, self.policy.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        grad_vector_product = torch.sum(kl_grad_vector * vector)
        grad_grad = torch.autograd.grad(
            grad_vector_product, self.policy.parameters())
        fisher_vector_product = torch.cat(
            [grad.contiguous().view(-1) for grad in grad_grad])
        return fisher_vector_product + (self.cg_damping * vector)

    @abc.abstractmethod
    def mean_kl_divergence(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def learn(self):
        self.sample_batch()
        # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
        imp_fac = self.compute_imp_fac(self.policy)
        # values and advantages are all 2-D Tensor. size: r.size(0) x 1
        self.estimate_value()
        # update policy
        if self.A is not None:
            self.loss = - (imp_fac * self.A.squeeze()).mean()
            # update value
            self.update_value()
        else:
            self.loss = - (imp_fac * self.V.squeeze()).mean()
        self.policy.zero_grad()
        loss_grad = torch.autograd.grad(
            self.loss, self.policy.parameters(), create_graph=True)
        # loss_grad_vector is a 1-D Variable including all parameters in self.policy
        loss_grad_vector = parameters_to_vector([grad for grad in loss_grad])
        # solve Ax = -g, A is Hessian Matrix of KL divergence
        trpo_grad_direc = self.conjunction_gradient(- loss_grad_vector)
        shs = .5 * torch.sum(trpo_grad_direc * self.hessian_vector_product(trpo_grad_direc))
        beta = torch.sqrt(self.max_kl / shs)
        fullstep = trpo_grad_direc * beta
        thetanew = parameters_to_vector(self.policy.parameters()) + fullstep
        vector_to_parameters(thetanew, self.policy.parameters())
        self.learn_step_counter += 1

class NPG_Gaussian(NPG, PG_Gaussian):
    def __init__(self,hyperparams):
        super(NPG_Gaussian, self).__init__(hyperparams)

    def mean_kl_divergence(self):
        mu1, sigma1 = self.policy(self.s)
        # 1e-8 make the final derivative not equal 0
        mu2 = Variable(mu1.data)
        sigma2 = Variable(sigma1.data)
        det1 = torch.cumprod(sigma1,dim = 1)[:,sigma1.size(1)-1]
        det2 = torch.cumprod(sigma2,dim = 1)[:,sigma2.size(1)-1]
        kl = 0.5 * (torch.log(det1) - torch.log(det2) - self.n_actions + torch.sum(sigma2 / sigma1, dim = 1) + torch.sum(torch.pow((mu1 - mu2),2) / sigma1,1))
        return kl.mean()

class NPG_Softmax(NPG,PG_Softmax):
    def __init__(self,hyperparams):
        super(NPG_Softmax, self).__init__(hyperparams)

    def mean_kl_divergence(self):
        distri1 = self.policy(self.s)
        distri2 = Variable(distri1.data)
        logratio = torch.log(distri2/distri1)
        kl = torch.sum(distri2 * logratio , 1)
        return kl.mean()