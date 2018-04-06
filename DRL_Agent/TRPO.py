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
from config import TRPO_CONFIG
from ApproFunc.TRPO import FCPOLICYTRPO, FCVALUETRPO
torch.set_default_tensor_type('torch.DoubleTensor')

class TRPO(Agent):
    def __init__(self,hyperparams):
        config = copy.deepcopy(TRPO_CONFIG)
        config.update(hyperparams)
        super(TRPO, self).__init__(config)
        self.action_bounds = config['action_bounds']
        # hyperparams for conjunction gradients
        self.cg_iters = config['cg_iters']
        self.cg_residual_tol = config['cg_residual_tol']
        self.cg_damping = config['cg_damping']
        self.policy_type = config['policy_type']
        self.value_type = config['value_type']
        self.max_kl = config['max_kl_divergence']
        self.memory = torch.Tensor(np.zeros((self.memory_size, self.n_features * 2 + 2 + 3 * self.n_actions)))
        # initial policy and value models
        if self.policy_type == 'FC':
            self.policy = FCPOLICYTRPO(self.n_features,    # input dim
                 self.n_actions,   # output dim
                 2,
                 outactive = F.tanh,
                 outscaler = self.action_bounds
                 )
        if self.value_type == 'FC':
            self.value = FCVALUETRPO(self.n_features,    # input dim
                 )
        self.loss_func_v = config['loss_func_v']()
        if config['v_optimizer'] == optim.LBFGS:
            self.using_lbfgs_for_V = True
        else:
            self.using_lbfgs_for_V = False
            self.v_optimizer = config['v_optimizer'](self.value.parameters(), lr = self.lr)

    def choose_action(self, s):
        s = Variable(torch.Tensor(s))
        mu,sigma = self.policy(s)
        a = torch.normal(mu,sigma)
        return np.array(a.data), np.array(mu.data), np.array(sigma.data)

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

    def mean_kl_divergence(self):
        mu1, sigma1 = self.policy(self.s)
        # 1e-8 make the final derivative not equal 0
        mu2 = Variable(mu1.data)
        sigma2 = Variable(sigma1.data)
        det1 = torch.cumprod(sigma1,dim = 1)[:,sigma1.size(1)-1]
        det2 = torch.cumprod(sigma2,dim = 1)[:,sigma2.size(1)-1]
        kl = 0.5 * (torch.log(det1) - torch.log(det2) - self.n_actions + torch.sum(sigma2 / sigma1, dim = 1) + torch.sum(torch.pow((mu1 - mu2),2) / sigma1,1))
        return kl.mean()

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

    def compute_V_A(self):
        masks = 1-self.done
        values = self.value(self.s)
        returns = Variable(torch.zeros(self.r.size(0), 1))
        advantages = Variable(torch.zeros(self.r.size(0), 1))
        prev_return = 0
        for i in reversed(range(self.r.size(0))):
            returns[i] = self.r[i] + self.gamma * prev_return * masks[i]
            advantages[i] = returns[i] - values[i]
            prev_return = float(returns[i, 0])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # values and advantages are all 2-D Tensor. size: r.size(0) x 1
        return returns, advantages

    def compute_imp_fac(self,model):
        # theta is the vectorized model parameters
        mucur, sigmacur = model(self.s)
        # important sampling coefficients
        # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
        imp_fac = torch.exp (self.compute_logp(mucur, sigmacur, self.a) - self.compute_logp(self.mu, self.sigma, self.a))
        return imp_fac

    def object_loss(self, theta):
        model = copy.deepcopy(self.policy)
        vector_to_parameters(theta, model.parameters())
        imp_fac = self.compute_imp_fac(model)
        V, A = self.compute_V_A()
        # this normalization is borrowed. I don't know why
        loss = -(A.squeeze() * imp_fac).mean()
        return loss

    def linear_search(self,x, fullstep, expected_improve_rate):
        accept_ratio = .1
        max_backtracks = 10
        fval = self.object_loss(x)
        for (_n_backtracks, stepfrac) in enumerate(list(.5 ** torch.range(0, max_backtracks-1))):
            #print("Search number {}...".format(_n_backtracks + 1))
            xnew = x + stepfrac * fullstep
            newfval = self.object_loss(xnew)
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve
            if ratio.data[0] > accept_ratio and actual_improve.data[0] > 0:
                return xnew
        return x

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

    def learn(self):
        self.s = Variable(self.memory[:, :self.n_features])
        self.a = Variable(self.memory[:, 2 * self.n_actions + self.n_features: 3 * self.n_actions + self.n_features])
        self.mu = Variable(self.memory[:, self.n_features:(self.n_actions + self.n_features)])
        self.sigma = Variable(self.memory[:, self.n_actions + self.n_features: 2 * self.n_actions + self.n_features])
        self.r = Variable(self.memory[:, (self.n_features + 3 * self.n_actions):(self.n_features + 3 * self.n_actions + 1)])
        self.done = Variable(self.memory[:, (self.n_features + 3 * self.n_actions + 1):(self.n_features + 3 * self.n_actions + 2)])
        # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
        imp_fac = self.compute_imp_fac(self.policy)
        # values and advantages are all 2-D Tensor. size: r.size(0) x 1
        V, A = self.compute_V_A()
        self.loss = -(A.squeeze() * imp_fac).mean()
        self.policy.zero_grad()
        loss_grad = torch.autograd.grad(
            self.loss, self.policy.parameters(), create_graph=True)
        # loss_grad_vector is a 1-D Variable including all parameters in self.policy
        loss_grad_vector = parameters_to_vector([grad for grad in loss_grad])
        # solve Ax = -g, A is Hessian Matrix of KL divergence
        trpo_grad_direc = self.conjunction_gradient( - loss_grad_vector)
        shs = .5 * torch.sum(trpo_grad_direc * self.hessian_vector_product(trpo_grad_direc))
        beta = torch.sqrt(self.max_kl / shs)
        fullstep = trpo_grad_direc * beta
        gdotstepdir = -torch.sum(loss_grad_vector * trpo_grad_direc)
        theta = self.linear_search(parameters_to_vector(
            self.policy.parameters()), fullstep, gdotstepdir * beta)
        # update value
        V_target = V
        if self.using_lbfgs_for_V:
            self.optim_value_lbfgs(V_target)
        else:
            s = Variable(self.memory[:, :self.n_features])
            V_eval = self.value(s)
            self.loss_v = self.loss_func_v(V_eval,V_target)
            self.value.zero_grad()
            self.loss_v.backward()
            self.v_optimizer.step()
        # update policy
        vector_to_parameters(theta, self.policy.parameters())
