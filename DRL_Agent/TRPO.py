from DRL_Agent.NPG import NPG, NPG_Gaussian, NPG_Softmax
import torch
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
import copy
from config import TRPO_CONFIG
import abc
torch.set_default_tensor_type('torch.DoubleTensor')

class TRPO(NPG):
    __metaclass__ = abc.ABCMeta
    def __init__(self, hyperparams):
        config = copy.deepcopy(TRPO_CONFIG)
        config.update(hyperparams)
        super(TRPO, self).__init__(config)
        self.accept_ratio = config['accept_ratio']
        self.max_search_num = config['max_search_num']
        self.step_frac = config['step_frac']

    def compute_imp_fac_other(self, model, using_batch = False):
        # theta is the vectorized model parameters
        if using_batch:
            mucur, sigmacur = model(self.s_batch)
            # important sampling coefficients
            # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
            imp_fac = torch.exp(
                self.compute_logp(mucur, sigmacur, self.a_batch) - self.compute_logp(self.mu_batch, self.sigma_batch,self.a_batch))
        else:
            mucur, sigmacur = model(self.s)
            # important sampling coefficients
            # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
            imp_fac = torch.exp(
                self.compute_logp(mucur, sigmacur, self.a) - self.compute_logp(self.mu, self.sigma, self.a))
        return imp_fac

    def object_loss(self, theta):
        model = copy.deepcopy(self.policy)
        vector_to_parameters(theta, model.parameters())
        imp_fac = self.compute_imp_fac_other(model)
        # this normalization is borrowed. I don't know why
        if self.A is not None:
            loss = - (imp_fac * self.A.squeeze()).mean()
        else:
            loss = - (imp_fac * self.V.squeeze()).mean()
        return loss

    def linear_search(self,x, fullstep, expected_improve_rate):
        accept_ratio = self.accept_ratio
        max_backtracks = self.max_search_num
        fval = self.object_loss(x)
        for (_n_backtracks, stepfrac) in enumerate(list(self.step_frac ** torch.arange(0, max_backtracks))):
            #print("Search number {}...".format(_n_backtracks + 1))
            xnew = x + stepfrac * fullstep
            newfval = self.object_loss(xnew)
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve
            if ratio.data[0] > accept_ratio and actual_improve.data[0] > 0:
                return xnew
        return x

    def learn(self):
        self.sample_batch()
        # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
        imp_fac = self.compute_imp_fac()
        self.estimate_value()
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
        trpo_grad_direc = self.conjunction_gradient( - loss_grad_vector)
        shs = .5 * torch.sum(trpo_grad_direc * self.hessian_vector_product(trpo_grad_direc))
        beta = torch.sqrt(self.max_kl / shs)
        fullstep = trpo_grad_direc * beta
        gdotstepdir = -torch.sum(loss_grad_vector * trpo_grad_direc)
        theta = self.linear_search(parameters_to_vector(
            self.policy.parameters()), fullstep, gdotstepdir * beta)
        # update policy
        vector_to_parameters(theta, self.policy.parameters())
        self.learn_step_counter += 1

class TRPO_Gaussian(NPG_Gaussian, TRPO):
    def __init__(self,hyperparams):
        super(TRPO_Gaussian, self).__init__(hyperparams)

class TRPO_Softmax(NPG_Softmax, TRPO):
    def __init__(self,hyperparams):
        super(TRPO_Softmax, self).__init__(hyperparams)


