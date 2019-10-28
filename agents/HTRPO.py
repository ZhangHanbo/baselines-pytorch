import torch
from torch import nn
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
import copy
from .TRPO import TRPO, TRPO_Gaussian, TRPO_Softmax
from .config import HTRPO_CONFIG
import abc
import numpy as np
from utils.mathutils import explained_variance
from collections import deque
import time
import copy
from utils.vecenv import space_dim
import time
from utils.rms import RunningMeanStd
import pickle
import os

class HTRPO(TRPO):
    __metaclass__ = abc.ABCMeta
    def __init__(self, hyperparams):
        config = copy.deepcopy(HTRPO_CONFIG)
        config.update(hyperparams)
        config['v_loss_reduction'] = 'none'
        super(HTRPO, self).__init__(config)
        self.sampled_goal_num = config['sampled_goal_num']
        self.goal_space = config['goal_space']
        self.d_goal = space_dim(self.goal_space)
        self.per_decision = config['per_decision']
        self.weight_is = config['weighted_is']
        assert self.goal_space, "No goal sampling space is specified."
        self.using_active_goals = config['using_active_goals']
        self.env = config['env']
        self.max_steps = self.env.max_episode_steps
        self.using_original_data = config['using_original_data']
        self.using_kl2 = config['using_kl2']
        self.n_valid_ep = 0

        self.norm_ob = config['norm_ob']
        if self.norm_ob:
            self.ob_rms = {}
            for key in self.env.observation_space.spaces.keys():
                self.ob_rms[key] = RunningMeanStd(shape=self.env.observation_space.spaces[key].shape)
            self.ob_mean = [0,]
            self.ob_var = [1,]

        assert config['sampled_goal_num'] is None \
               or config['sampled_goal_num'] > 0 \
               or config['using_original_data'], 'Data type must be specified.'
        self.n_traj = 0

        # HPG as the baseline of our HTRPO algorithm. HPG: http://arxiv.org/pdf/1711.06006
        self.using_hpg = config['using_hpg']
        if self.using_hpg:
            if self.mom is not None:
                self.optimizer = self.optimizer_type(self.policy.parameters(), lr=self.lr, momontum=self.mom)
            else:
                self.optimizer = self.optimizer_type(self.policy.parameters(), lr=self.lr)

    def generate_subgoals(self):
        if not self.using_active_goals:
            # generate subgoals randomly
            # TODO: SUPPORT GOAL SAMPLING
            self.subgoals = None
        else:
            # generate subgoals from sampled data
            ags = self.achieved_goal.cpu().numpy()
            # self.subgoals = np.unique(ags, axis=0)
            self.subgoals = np.unique(ags.round(decimals=2), axis=0)
            data_size = self.subgoals.shape[0]
            if self.sampled_goal_num is not None:
                size = min(self.sampled_goal_num, data_size)
            else:
                size = data_size
            indices = np.random.choice(data_size, size, False)
            self.subgoals = self.subgoals[indices]

    @abc.abstractmethod
    def generate_fake_data(self):
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def split_episode(self):
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def reset_training_data(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def save_model(self, save_path):
        if self.norm_ob:
            with open(os.path.join(save_path, 'normalizer' + str(self.learn_step_counter) + '.pkl'), 'wb') as f:
                pickle.dump(self.ob_rms, f)
        TRPO.save_model(self, save_path)

    def load_model(self, load_path, load_point):
        if self.norm_ob:
            with open(os.path.join(load_path, 'normalizer' + str(load_point) + '.pkl'), 'rb') as f:
                self.ob_rms = pickle.load(f)
                self.ob_mean = np.concatenate((self.ob_rms['observation'].mean, self.ob_rms['desired_goal'].mean),
                                              axis=-1)
                self.ob_var = np.concatenate((self.ob_rms['observation'].var, self.ob_rms['desired_goal'].var), axis=-1)
        TRPO.load_model(self, load_path, load_point)

    def learn(self):
        self.n_traj = 0
        if self.using_hpg:
            return self.learn_hpg()
        elif self.sampled_goal_num is None or self.sampled_goal_num > 0:
            return self.learn_htrpo()
        else:
            return self.learn_trpo()

    def learn_hpg(self):
        self.sample_batch()
        self.split_episode()
        # TODO using reasonable subgoals instead of random ones
        self.generate_subgoals()
        if not self.using_original_data:
            self.reset_training_data()
        if self.sampled_goal_num is None or self.sampled_goal_num > 0 :
            self.generate_fake_data()

        if self.norm_ob:
            self.ob_rms['observation'].update(self.s.cpu().numpy())
            self.ob_rms['desired_goal'].update(self.goal.cpu().numpy())
            self.ob_mean = np.concatenate((self.ob_rms['observation'].mean, self.ob_rms['desired_goal'].mean), axis=-1)
            self.ob_var = np.concatenate((self.ob_rms['observation'].var, self.ob_rms['desired_goal'].var), axis=-1)

        self.s = torch.cat((self.s, self.goal), dim=1)
        self.s_ = torch.cat((self.s_, self.goal), dim=1)

        if self.norm_ob:
            self.s = torch.clamp((self.s - torch.Tensor(self.ob_mean).type_as(self.s).unsqueeze(0)) / torch.sqrt(
                torch.clamp(torch.Tensor(self.ob_var), 1e-4).type_as(self.s).unsqueeze(0)), -5, 5)

        self.estimate_value()
        if self.value_type is not None:
            # update value
            for i in range(self.iters_v):
                self.update_value()

        # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
        # Likelihood Ratio
        imp_fac = self.compute_imp_fac()
        # update policy (old value estimator)
        # self.loss = - (imp_fac * self.gamma_discount * self.hratio * self.A.squeeze()).sum()

        # update policy
        if self.value_type:
            # old value estimator
            self.A = self.gamma_discount * self.hratio * self.A
        else:
            self.A = self.gamma_discount * self.A
        self.loss = - (imp_fac * self.A).sum() / self.n_traj

        self.policy.zero_grad()
        self.loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.learn_step_counter += 1
        self.cur_kl = self.mean_kl_divergence().item()
        self.policy_ent = self.compute_entropy().item()

    def learn_trpo(self):
        self.sample_batch()
        self.split_episode()

        if self.norm_ob:
            self.ob_rms['observation'].update(self.s.cpu().numpy())
            self.ob_rms['desired_goal'].update(self.goal.cpu().numpy())
            self.ob_mean = np.concatenate((self.ob_rms['observation'].mean, self.ob_rms['desired_goal'].mean), axis=-1)
            self.ob_var = np.concatenate((self.ob_rms['observation'].var, self.ob_rms['desired_goal'].var), axis=-1)

        self.s = torch.cat((self.s, self.goal), dim=1)
        self.s_ = torch.cat((self.s_, self.goal), dim=1)

        if self.norm_ob:
            self.s = torch.clamp((self.s - torch.Tensor(self.ob_mean).type_as(self.s).unsqueeze(0)) / torch.sqrt(
                torch.clamp(torch.Tensor(self.ob_var), 1e-4).type_as(self.s).unsqueeze(0)), -5, 5)

        # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
        imp_fac = self.compute_imp_fac()
        self.estimate_value()
        self.A = (self.A - self.A.mean()) / (self.A.std() + 1e-8)
        self.loss = - (imp_fac * self.A).mean() - self.entropy_weight * self.compute_entropy()
        if self.value_type is not None:
            # update value
            for i in range(self.iters_v):
                self.update_value()
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
        self.cur_kl = self.mean_kl_divergence().item()
        self.policy_ent = self.compute_entropy().item()

    def learn_htrpo(self):
        b_t = time.time()
        self.sample_batch()
        self.split_episode()

        self.generate_subgoals()
        if not self.using_original_data:
            self.reset_training_data()
        if self.sampled_goal_num is None or self.sampled_goal_num > 0:
            self.generate_fake_data()

        if self.norm_ob:
            self.ob_rms['observation'].update(self.s.cpu().numpy())
            self.ob_rms['desired_goal'].update(self.goal.cpu().numpy())
            self.ob_mean = np.concatenate((self.ob_rms['observation'].mean, self.ob_rms['desired_goal'].mean), axis=-1)
            self.ob_var = np.concatenate((self.ob_rms['observation'].var, self.ob_rms['desired_goal'].var), axis=-1)

        self.s = torch.cat((self.s, self.goal), dim=1)
        self.s_ = torch.cat((self.s_, self.goal), dim=1)

        if self.norm_ob:
            self.s = torch.clamp((self.s - torch.Tensor(self.ob_mean).type_as(self.s).unsqueeze(0)) / torch.sqrt(
                torch.clamp(torch.Tensor(self.ob_var), 1e-4).type_as(self.s).unsqueeze(0)), -5, 5)

        # Optimize Value Estimator
        self.estimate_value()
        if self.value_type is not None:
            # update value
            for i in range(self.iters_v):
                self.update_value()

        # Optimize Policy
        # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
        # Likelihood Ratio
        # self.estimate_value()
        imp_fac = self.compute_imp_fac()

        if self.value_type:
            # old value estimator
            self.A = self.gamma_discount * self.hratio * self.A
        else:
            self.A = self.gamma_discount * self.A

        # Here mean() and sum() / self.n_traj is equivalent, because there
        # is only a coefficient between two expressions. This coefficient
        # will be compensated by the stepsize computation in TRPO. However,
        # in vanilla PG, there is no compensation, therefore, it needs to
        # be in the exact form of the euqation in the paper.
        self.loss = - (imp_fac * self.A).mean() - self.entropy_weight * self.compute_entropy()

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
        gdotstepdir = -torch.sum(loss_grad_vector * trpo_grad_direc)
        theta = self.linear_search(parameters_to_vector(
            self.policy.parameters()), fullstep, gdotstepdir * beta)
        vector_to_parameters(theta, self.policy.parameters())
        self.learn_step_counter += 1
        self.cur_kl = self.mean_kl_divergence().item()
        self.policy_ent = self.compute_entropy().item()
        print("iteration time:   {:.4f}".format(time.time()-b_t))

    def update_value(self, inds = None):
        if inds is None:
            inds = np.arange(self.s.size(0))
        V_target = self.esti_R[inds]
        assert not self.using_lbfgs_for_V, "LBFGS is not supported by HTRPO."

        V_eval = self.value(self.s[inds]).squeeze()
        self.loss_v = self.loss_func_v(V_eval, V_target).mean()
        self.value_loss = self.loss_v.item()
        self.value.zero_grad()
        self.loss_v.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
        self.v_optimizer.step()

    def estimate_value(self):
        # undone but the final state.
        fake_done = torch.nonzero(self.done.squeeze() == 2).squeeze(-1)
        # make all fake_done equal to 1
        self.done[self.done == 2] = 1
        returns = torch.zeros(self.r.size(0), 1).type_as(self.s)
        prev_return = 0

        # TODO: Improve HGAE
        # if self.value_type is not None:
        #     values = self.value(self.s)
        #     delta = torch.zeros(self.r.size(0), 1).type_as(self.s)
        #     advantages = torch.zeros(self.r.size(0), 1).type_as(self.s)
        #
        #     # mask is a 1-d vector, therefore, e_points is also a 1-d vector
        #     e_points = torch.nonzero(self.done.squeeze() == 1).squeeze()
        #     b_points = - torch.ones(size = e_points.size()).type_as(e_points)
        #     b_points[1:] = e_points[:-1]
        #     ep_lens = e_points - b_points
        #     assert ep_lens.min().item() > 0, "Some episode lengths are smaller than 0."
        #     max_len = torch.max(ep_lens).item()
        #     uncomplete_flag = ep_lens > 0
        #
        #     delta[e_points[ep_lens > 1]] = r_h[e_points[ep_lens > 1]] - \
        #                                    values[e_points[ep_lens > 1]] * hratio[e_points[ep_lens > 1] - 1]
        #     delta[e_points[ep_lens == 1]] = r_h[e_points[ep_lens == 1]] - values[e_points[ep_lens == 1]]
        #     if fake_done.numel() > 0:
        #         delta[fake_done] += self.value(self.s_[fake_done]).resize_as(delta[fake_done]) * hratio[fake_done]
        #     advantages[e_points] = delta[e_points]
        #     returns[e_points] = r_h[e_points]
        #
        #     for i in range(1, max_len):
        #         uncomplete_flag[ep_lens <= i] = 0
        #         inds = (e_points - i)[uncomplete_flag]
        #         epls = ep_lens[uncomplete_flag]
        #         delta[inds[epls > 1]] = r_h[inds[epls > 1]] + self.gamma * values[inds[epls > 1] + 1] * hratio[
        #             inds[epls > 1]] - values[inds[epls > 1]] * hratio[inds[epls > 1] - 1]
        #         delta[inds[epls == 1]] = r_h[inds[epls == 1]] + self.gamma * values[inds[epls == 1] + 1] * hratio[
        #             inds[epls == 1]] - values[inds[epls == 1]]
        #         advantages[inds] = delta[inds] + self.gamma * self.lamb * advantages[inds + 1]
        #         returns[inds] = r_h[inds] + self.gamma * returns[inds + 1]
        #
        #     # TODO: Using HGAE for value target computation instead of TD(0)
        #     esti_return = self.r + self.value(self.s_)
        #     esti_return[e_points] = self.r[e_points]
        #
        #     # values returns advantages and estimated returns
        #     self.V = values.squeeze().detach()
        #     self.R = returns.squeeze().detach()
        #     self.A = advantages.squeeze().detach()
        #     self.esti_R = esti_return.squeeze().detach()

        # using value approximator
        if self.value_type is not None:
            values = self.value(self.s)
            delta = torch.zeros(self.r.size(0), 1).type_as(self.s)
            advantages = torch.zeros(self.r.size(0), 1).type_as(self.s)

            # mask is a 1-d vector, therefore, e_points is also a 1-d vector
            e_points = torch.nonzero(self.done.squeeze() == 1).squeeze()
            b_points = - torch.ones(size = e_points.size()).type_as(e_points)
            b_points[1:] = e_points[:-1]
            ep_lens = e_points - b_points
            assert ep_lens.min().item() > 0, "Some episode lengths are smaller than 0."
            max_len = torch.max(ep_lens).item()
            uncomplete_flag = ep_lens > 0

            delta[e_points] = self.r[e_points] - values[e_points]
            if fake_done.numel() > 0:
                delta[fake_done] += self.value(self.s_[fake_done]).resize_as(delta[fake_done])
            advantages[e_points] = delta[e_points]
            returns[e_points] = self.r[e_points]
            for i in range(1, max_len):
                uncomplete_flag[ep_lens <= i] = 0
                # TD-error
                inds = (e_points - i)[uncomplete_flag]
                delta[inds] = self.r[inds] + self.gamma * values[inds + 1] - values[inds]
                advantages[inds] = delta[inds] + self.gamma * self.lamb * advantages[inds + 1]
                returns[inds] = self.r[inds] + self.gamma * returns[inds + 1]

            # Estimated Return, from OpenAI baseline.
            esti_return = values + advantages
            # values returns advantages and estimated returns
            self.V = values.squeeze().detach()
            self.R = returns.squeeze().detach()
            self.A = advantages.squeeze().detach()
            self.A = (self.A - self.A.mean()) / (self.A.std() + 1e-10)
            self.esti_R = esti_return.squeeze().detach()

        else:
            # For Hindsight ratio used in this function, it should be 2-d with a second redundant dimension.
            hratio = self.hratio.unsqueeze(1)
            # Reward with hindsight ratio weights.
            r_h = self.r * hratio
            e_points = torch.nonzero(self.done.squeeze() == 1).squeeze()
            b_points = - torch.ones(size = e_points.size()).type_as(e_points)
            b_points[1:] = e_points[:-1]
            ep_lens = e_points - b_points
            assert ep_lens.min().item() > 0, "Some episode lengths are smaller than 0."
            max_len = torch.max(ep_lens).item()
            uncomplete_flag = ep_lens > 0
            returns[e_points] = r_h[e_points]
            for i in range(1, max_len):
                uncomplete_flag[ep_lens <= i] = 0
                inds = (e_points - i)[uncomplete_flag]
                returns[inds] = r_h[inds] + self.gamma * returns[inds + 1]

            self.R = returns.squeeze().detach()
            # Here self.A is actually not advantages. It works for policy updates, which
            # means that it is used to measure how good a action is.
            self.A = self.R

class HTRPO_Gaussian(HTRPO, TRPO_Gaussian):
    def __init__(self, hyperparams):
        super(HTRPO_Gaussian, self).__init__(hyperparams)

    def choose_action(self, s, greedy = False):
        if self.norm_ob:
            s = torch.clamp((s - torch.Tensor(self.ob_mean).type_as(s).unsqueeze(0)) / torch.sqrt(
                torch.clamp(torch.Tensor(self.ob_var), 1e-4).type_as(s).unsqueeze(0)), -5, 5)
        return TRPO_Gaussian.choose_action(self, s, greedy)

    def generate_fake_data(self):
        self.subgoals = torch.Tensor(self.subgoals).type_as(self.s)
        # number of subgoals
        n_g = self.subgoals.shape[0]

        # for weighted importance sampling, Ne x Ng x T
        h_ratios = torch.zeros(size = (len(self.episodes), n_g, self.max_steps)).type_as(self.s)
        h_ratios_mask = torch.zeros(size = (len(self.episodes), n_g, self.max_steps)).type_as(self.s)
        for ep in range(len(self.episodes)):
            # original episode length
            ep_len = self.episodes[ep]['length']

            # Modify episode length and rewards.
            # Ng x T
            r_f = self.env.compute_reward(self.episodes[ep]['achieved_goal'].unsqueeze(0).repeat(n_g,1,1).cpu().numpy(),
                                          self.subgoals.unsqueeze(1).repeat(1,ep_len,1).cpu().numpy(), None)
            # Here, reward will be 0 when the goal is not achieved, else 1.
            r_f += 1
            # For negative episode, there is no positive reward, all are 0.
            neg_ep_inds = np.where(r_f.sum(axis=-1) == 0)

            # In reward, there are only 0 and 1. The first 1's position indicates the episode length.
            l_f = np.argmax(r_f, axis=-1)
            l_f += 1
            # For all negative episodes, the length is the value of max_steps.
            l_f[neg_ep_inds] = ep_len
            # lengths: Ng
            l_f = torch.Tensor(l_f).type_as(self.s).long()

            # Ng x T
            mask = torch.Tensor(np.arange(1, ep_len + 1)).type_as(self.s).repeat(n_g, 1)
            mask[mask > l_f.type_as(self.s).unsqueeze(1)] = 0
            mask[mask > 0] = 1
            # filter out the episodes where at beginning, the goal is achieved.
            mask[l_f == 1] = 0

            r_f = torch.Tensor(r_f).type_as(self.r)

            # Rewards are 0 and T - t_done + 1
            # Ng x T
            r_f[range(r_f.size(0)), l_f - 1] = (self.max_steps - l_f + 1).type_as(self.r)
            r_f[neg_ep_inds] = 0

            d_f = self.episodes[ep]['done'].squeeze().repeat(n_g, 1)
            d_f[range(d_f.size(0)), l_f - 1] = 1

            h_ratios_mask[ep][:,:ep_len] = mask

            expanded_s = self.episodes[ep]['s'][:ep_len].repeat(n_g, 1)
            expanded_g = self.subgoals.unsqueeze(1).repeat(1, ep_len, 1).reshape(-1, self.d_goal)
                         # - self.episodes[ep]['achieved_goal'].unsqueeze(0).repeat(n_g,1,1).reshape(-1, self.d_goal)
            # Ng * T x (Ds + Dg)
            fake_input = torch.cat((expanded_s, expanded_g), dim = 1)
            if self.norm_ob:
                fake_input = torch.clamp((fake_input - torch.Tensor(self.ob_mean).type_as(self.s).unsqueeze(0)) / torch.sqrt(
                    torch.clamp(torch.Tensor(self.ob_var), 1e-4).type_as(self.s).unsqueeze(0)), -5, 5)

            fake_mu, fake_logsigma, fake_sigma = self.policy(fake_input)
            fake_mu = fake_mu.detach()
            fake_sigma = fake_sigma.detach()
            fake_logsigma = fake_logsigma.detach()

            # Ng * T x Da
            expanded_a = self.episodes[ep]['a'].repeat(n_g, 1)

            # Ng x T
            fake_logpac = self.compute_logp(fake_mu, fake_logsigma, fake_sigma, expanded_a).reshape(n_g, ep_len)
            expanded_logpac_old = self.episodes[ep]['logpac_old'].repeat(n_g,1).reshape(n_g, -1)
            d_logp = fake_logpac - expanded_logpac_old

            # generate hindsight ratio
            # Ng x T
            if self.per_decision:
                h_ratio = torch.exp(d_logp.cumsum(dim=1)) + 1e-10
                h_ratio *= mask
                h_ratios[ep][:, :ep_len] = h_ratio
            else:
                h_ratio = torch.exp(torch.sum(d_logp, keepdim=True)).repeat(1, ep_len) + 1e-10
                h_ratio *= mask
                h_ratios[ep][:, :ep_len] = h_ratio

            # make all data one batch
            mask = mask.reshape(-1) > 0
            self.s = torch.cat((self.s, expanded_s[mask]), dim=0)
            self.s_ = torch.cat((self.s_, self.episodes[ep]['s_'].repeat(n_g, 1)[mask]), dim=0)
            self.a = torch.cat((self.a, expanded_a[mask]), dim=0)
            self.goal = torch.cat((self.goal, expanded_g[mask]), dim=0)
            self.mu = torch.cat((self.mu, fake_mu[mask]), dim=0)
            self.sigma = torch.cat((self.sigma, fake_sigma[mask]), dim=0)

            self.r = torch.cat((self.r, r_f.reshape(n_g * ep_len, 1)[mask]), dim=0)
            self.done = torch.cat((self.done, d_f.reshape(n_g * ep_len, 1)[mask]), dim=0)
            self.logpac_old = torch.cat((self.logpac_old, fake_logpac.reshape(n_g * ep_len, 1)[mask]), dim=0)

            # Ng x T
            gamma_discount = torch.pow(self.gamma, torch.Tensor(np.arange(1, ep_len + 1)).type_as(self.s)).repeat(n_g, 1)
            self.gamma_discount = torch.cat((self.gamma_discount, gamma_discount.reshape(n_g * ep_len)[mask]), dim=0)

            self.n_traj += n_g

        if self.weight_is:
            h_ratios_sum = torch.sum(h_ratios, dim=0, keepdim = True)
            h_ratios /= h_ratios_sum

        h_ratios_mask = h_ratios_mask.reshape(-1) > 0
        self.hratio = torch.cat((self.hratio, h_ratios.reshape(-1)[h_ratios_mask]), dim=0)

    def reset_training_data(self):
        self.s = torch.Tensor(size = (0,) + self.s.size()[1:]).type_as(self.s)
        self.s_ = torch.Tensor(size = (0,) + self.s_.size()[1:]).type_as(self.s_)
        self.a = torch.Tensor(size = (0,) + self.a.size()[1:]).type_as(self.a)
        self.r = torch.Tensor(size = (0,) + self.r.size()[1:]).type_as(self.r)
        self.done = torch.Tensor(size = (0,) + self.done.size()[1:]).type_as(self.done)
        self.goal = torch.Tensor(size = (0,) + self.goal.size()[1:]).type_as(self.goal)
        self.gamma_discount = torch.Tensor(size = (0,) + self.gamma_discount.size()[1:]).type_as(self.gamma_discount)
        self.hratio = torch.Tensor(size = (0,) + self.hratio.size()[1:]).type_as(self.hratio)
        self.logpac_old = torch.Tensor(size = (0,) + self.logpac_old.size()[1:]).type_as(self.logpac_old)
        self.mu = torch.Tensor(size = (0,) + self.mu.size()[1:]).type_as(self.mu)
        self.sigma = torch.Tensor(size = (0,) + self.sigma.size()[1:]).type_as(self.sigma)
        self.achieved_goal = torch.Tensor(size = (0,) + self.achieved_goal.size()[1:]).type_as(self.achieved_goal)
        self.desired_goal = torch.Tensor(size = (0,) + self.desired_goal.size()[1:]).type_as(self.desired_goal)
        self.n_traj = 0

    def split_episode(self):
        self.n_valid_ep = 0
        # Episodes store all the original data and will be used to generate fake
        # data instead of being used to train model
        assert self.other_data, "Hindsight algorithms need goal infos."
        self.desired_goal = self.other_data['desired_goal']
        self.achieved_goal = self.other_data['achieved_goal']
        self.goal = self.desired_goal
        # initialize real data's hindsight ratios.
        self.hratio = torch.ones(self.s.size(0)).type_as(self.s)

        self.episodes = []
        endpoints = (0,) + tuple((torch.nonzero(self.done[:, 0] > 0) + 1).squeeze().cpu().numpy().tolist())

        self.r += 1
        suc_poses = torch.nonzero(self.r==1)[:, 0]
        for suc_pos in suc_poses:
            temp = suc_pos - torch.Tensor(endpoints).type_as(self.r) + 1
            temp = temp[temp > 0]
            self.r[suc_pos] = self.max_steps - torch.min(temp) + 1

        # TODO: For gym envs, the episode will not end when the goal is achieved, deal with that!
        for i in range(len(endpoints) - 1):
            is_valid_ep = \
                np.unique(np.round(self.achieved_goal[endpoints[i]: endpoints[i + 1]].cpu().numpy(), decimals=2),
                      axis=0).shape[0] > 1
            self.n_valid_ep += is_valid_ep
            # if is_valid_ep:
            episode = {}
            episode['s'] = self.s[endpoints[i] : endpoints[i + 1]]
            episode['a'] = self.a[endpoints[i]: endpoints[i + 1]]
            episode['r'] = self.r[endpoints[i]: endpoints[i + 1]]
            episode['done'] = self.done[endpoints[i]: endpoints[i + 1]]
            episode['s_'] = self.s_[endpoints[i]: endpoints[i + 1]]
            episode['logpac_old'] = self.logpac_old[endpoints[i]: endpoints[i + 1]]
            episode['mu'] = self.mu[endpoints[i]: endpoints[i + 1]]
            episode['sigma'] = self.sigma[endpoints[i]: endpoints[i + 1]]
            episode['desired_goal'] = self.desired_goal[endpoints[i]: endpoints[i + 1]]
            episode['achieved_goal'] = self.achieved_goal[endpoints[i]: endpoints[i + 1]]

            episode['length'] = endpoints[i + 1] - endpoints[i]
            episode['gamma_discount'] = torch.pow(self.gamma, torch.Tensor(np.arange(episode['length'])).type_as(
                self.s)).unsqueeze(1)

            self.episodes.append(episode)

        # if len(self.episodes) > 0:
        self.gamma_discount = torch.cat([ep['gamma_discount'].squeeze(1) for ep in self.episodes], dim=0)
        self.n_traj += len(self.episodes)

    def mean_kl_divergence(self, inds = None, model = None):
        if inds is None:
            inds = np.arange(self.s.size(0))
        if model is None:
            model = self.policy
        mu1, logsigma1, sigma1 = model(self.s[inds])
        logp = self.compute_logp(mu1, logsigma1, sigma1, self.a[inds])
        logp_old = self.logpac_old[inds].squeeze()
        if self.using_kl2:
            mean_kl = (1 - self.gamma) * torch.sum(
                self.hratio[inds] * self.gamma_discount * 0.5 * torch.pow((logp_old - logp), 2)) / self.n_traj
        else:
            mean_kl = (1 - self.gamma) * torch.sum(
                self.hratio[inds] * self.gamma_discount * (logp_old - logp)) / self.n_traj

        return mean_kl

class HTRPO_Softmax(HTRPO, TRPO_Softmax):
    def __init__(self, hyperparams):
        super(HTRPO_Softmax, self).__init__(hyperparams)

    def choose_action(self, s, greedy = False):
        if self.norm_ob:
            s = torch.clamp((s - torch.Tensor(self.ob_mean).type_as(s).unsqueeze(0)) / torch.sqrt(
                torch.clamp(torch.Tensor(self.ob_var), 1e-4).type_as(s).unsqueeze(0)), -5, 5)
        return TRPO_Softmax.choose_action(self, s, greedy)

    def generate_fake_data(self):
        self.subgoals = torch.Tensor(self.subgoals).type_as(self.s)
        # number of subgoals
        n_g = self.subgoals.shape[0]

        # for weighted importance sampling, Ne x Ng x T
        h_ratios = torch.zeros(size = (len(self.episodes), n_g, self.max_steps)).type_as(self.s)
        h_ratios_mask = torch.zeros(size = (len(self.episodes), n_g, self.max_steps)).type_as(self.s)
        for ep in range(len(self.episodes)):
            # original episode length
            ep_len = self.episodes[ep]['length']

            # Modify episode length and rewards.
            # Ng x T
            r_f = self.env.compute_reward(
                self.episodes[ep]['achieved_goal'].unsqueeze(0).repeat(n_g, 1, 1).cpu().numpy(),
                self.subgoals.unsqueeze(1).repeat(1, ep_len, 1).cpu().numpy(), None)
            # Here, reward will be 0 when the goal is not achieved, else 1.
            r_f += 1
            # For negative episode, there is no positive reward, all are 0.
            neg_ep_inds = np.where(r_f.sum(axis=-1) == 0)

            # In reward, there are only 0 and 1. The first 1's position indicates the episode length.
            l_f = np.argmax(r_f, axis=-1)
            l_f += 1
            # For all negative episodes, the length is the value of max_steps.
            l_f[neg_ep_inds] = ep_len
            # lengths: Ng
            l_f = torch.Tensor(l_f).type_as(self.s).long()
            r_f = torch.Tensor(r_f).type_as(self.r)

            # Rewards are 0 and T - t_done + 1
            # Ng x T
            r_f[range(r_f.size(0)), l_f - 1] = (self.max_steps - l_f + 1).type_as(self.r)
            r_f[neg_ep_inds] = 0

            d_f = self.episodes[ep]['done'].squeeze().repeat(n_g, 1)
            d_f[range(d_f.size(0)), l_f - 1] = 1

            # Ng x T
            mask = torch.Tensor(np.arange(1, ep_len + 1)).type_as(self.s).repeat(n_g, 1)
            mask[mask > l_f.type_as(self.s).unsqueeze(1)] = 0
            mask[mask > 0] = 1
            # mask[l_f == 1] = 0
            h_ratios_mask[ep][:,:ep_len] = mask

            expanded_s = self.episodes[ep]['s'][:ep_len].repeat(n_g, 1)
            expanded_g = self.subgoals.unsqueeze(1).repeat(1, ep_len, 1).reshape(-1, self.d_goal)
                         # - self.episodes[ep]['achieved_goal'].unsqueeze(0).repeat(n_g,1,1).reshape(-1, self.d_goal)
            # Ng * T x (Ds + Dg)
            fake_input = torch.cat((expanded_s, expanded_g), dim = 1)
            if self.norm_ob:
                fake_input = torch.clamp((fake_input - torch.Tensor(self.ob_mean).type_as(self.s).unsqueeze(0)) / torch.sqrt(
                    torch.clamp(torch.Tensor(self.ob_var), 1e-4).type_as(self.s).unsqueeze(0)), -5, 5)

            # Ng * T x Na
            fake_distri = self.policy(fake_input).detach()

            # Ng * T x Da
            expanded_a = self.episodes[ep]['a'].repeat(n_g, 1)

            # Ng x T
            fake_logpac = self.compute_logp(fake_distri, expanded_a).reshape(n_g, ep_len)
            expanded_logpac_old = self.episodes[ep]['logpac_old'].repeat(n_g,1).reshape(n_g, -1)
            d_logp = fake_logpac - expanded_logpac_old

            # generate hindsight ratio
            # Ng x T
            if self.per_decision:
                h_ratio = torch.exp(d_logp.cumsum(dim=1)) + 1e-10
                h_ratio *= mask
                h_ratios[ep][:, :ep_len] = h_ratio
            else:
                h_ratio = torch.exp(torch.sum(d_logp, keepdim=True)).repeat(1, ep_len) + 1e-10
                h_ratio *= mask
                h_ratios[ep][:, :ep_len] = h_ratio

            # make all data one batch
            mask = mask.reshape(-1) > 0
            self.s = torch.cat((self.s, expanded_s[mask]), dim=0)
            self.s_ = torch.cat((self.s_, self.episodes[ep]['s_'].repeat(n_g, 1)[mask]), dim=0)
            self.a = torch.cat((self.a, expanded_a[mask]), dim=0)
            self.goal = torch.cat((self.goal, expanded_g[mask]), dim=0)
            self.distri = torch.cat((self.distri, fake_distri[mask]), dim=0)

            self.r = torch.cat((self.r, r_f.reshape(n_g * ep_len, 1)[mask]), dim=0)
            self.done = torch.cat((self.done, d_f.reshape(n_g * ep_len, 1)[mask]), dim=0)
            self.logpac_old = torch.cat((self.logpac_old, fake_logpac.reshape(n_g * ep_len, 1)[mask]), dim=0)

            # Ng x T
            gamma_discount = torch.pow(self.gamma, torch.Tensor(np.arange(1, ep_len + 1)).type_as(self.s)).repeat(n_g, 1)
            self.gamma_discount = torch.cat((self.gamma_discount, gamma_discount.reshape(n_g * ep_len)[mask]), dim=0)

            self.n_traj += n_g

        if self.weight_is:
            h_ratios_sum = torch.sum(h_ratios, dim=0, keepdim = True)
            h_ratios /= h_ratios_sum

        h_ratios_mask = h_ratios_mask.reshape(-1) > 0
        self.hratio = torch.cat((self.hratio, h_ratios.reshape(-1)[h_ratios_mask]), dim=0)

    def reset_training_data(self):
        self.s = torch.Tensor(size = (0,) + self.s.size()[1:]).type_as(self.s)
        self.s_ = torch.Tensor(size = (0,) + self.s_.size()[1:]).type_as(self.s_)
        self.a = torch.Tensor(size = (0,) + self.a.size()[1:]).type_as(self.a)
        self.r = torch.Tensor(size = (0,) + self.r.size()[1:]).type_as(self.r)
        self.done = torch.Tensor(size = (0,) + self.done.size()[1:]).type_as(self.done)
        self.goal = torch.Tensor(size = (0,) + self.goal.size()[1:]).type_as(self.goal)
        self.gamma_discount = torch.Tensor(size = (0,) + self.gamma_discount.size()[1:]).type_as(self.gamma_discount)
        self.hratio = torch.Tensor(size = (0,) + self.hratio.size()[1:]).type_as(self.hratio)
        self.logpac_old = torch.Tensor(size = (0,) + self.logpac_old.size()[1:]).type_as(self.logpac_old)
        self.distri = torch.Tensor(size = (0,) + self.distri.size()[1:]).type_as(self.distri)
        self.achieved_goal = torch.Tensor(size = (0,) + self.achieved_goal.size()[1:]).type_as(self.achieved_goal)
        self.desired_goal = torch.Tensor(size = (0,) + self.desired_goal.size()[1:]).type_as(self.desired_goal)
        self.n_traj = 0

    def split_episode(self):
        self.n_valid_ep = 0
        # Episodes store all the original data and will be used to generate fake
        # data instead of being used to train model
        assert self.other_data, "Hindsight algorithms need goal infos."
        self.desired_goal = self.other_data['desired_goal']
        self.achieved_goal = self.other_data['achieved_goal']
        self.goal = self.desired_goal
        # initialize real data's hindsight ratios.
        self.hratio = torch.ones(self.s.size(0)).type_as(self.s)

        self.episodes = []
        endpoints = (0,) + tuple((torch.nonzero(self.done[:, 0] > 0) + 1).squeeze().cpu().numpy().tolist())

        self.r += 1
        suc_poses = torch.nonzero(self.r==1)[:, 0]
        for suc_pos in suc_poses:
            temp = suc_pos - torch.Tensor(endpoints).type_as(self.r) + 1
            temp = temp[temp > 0]
            self.r[suc_pos] = self.max_steps - torch.min(temp) + 1

        for i in range(len(endpoints) - 1):

            self.n_valid_ep += \
                np.unique(np.round(self.achieved_goal[endpoints[i]: endpoints[i + 1]].cpu().numpy(), decimals=2),
                          axis=0).shape[0] > 1

            episode = {}
            episode['s'] = self.s[endpoints[i] : endpoints[i + 1]]
            episode['a'] = self.a[endpoints[i]: endpoints[i + 1]]
            episode['r'] = self.r[endpoints[i]: endpoints[i + 1]]
            episode['done'] = self.done[endpoints[i]: endpoints[i + 1]]
            episode['s_'] = self.s_[endpoints[i]: endpoints[i + 1]]
            episode['logpac_old'] = self.logpac_old[endpoints[i]: endpoints[i + 1]]
            episode['distri'] = self.distri[endpoints[i]: endpoints[i + 1]]
            episode['desired_goal'] = self.desired_goal[endpoints[i]: endpoints[i + 1]]
            episode['achieved_goal'] = self.achieved_goal[endpoints[i]: endpoints[i + 1]]

            episode['length'] = endpoints[i + 1] - endpoints[i]
            episode['gamma_discount'] = torch.pow(self.gamma, torch.Tensor(np.arange(episode['length'])).type_as(
                self.s)).unsqueeze(1)
            self.episodes.append(episode)

        self.gamma_discount = torch.cat([ep['gamma_discount'].squeeze(1) for ep in self.episodes], dim=0)
        self.n_traj += len(self.episodes)

    def mean_kl_divergence(self, inds = None, model = None):
        if inds is None:
            inds = np.arange(self.s.size(0))
        if model is None:
            model = self.policy
        distri = model(self.s[inds])
        logp = self.compute_logp(distri, self.a[inds])
        logp_old = self.logpac_old[inds].squeeze()
        if self.using_kl2:
            mean_kl = (1 - self.gamma) * torch.sum(
                self.hratio[inds] * self.gamma_discount * 0.5 * torch.pow((logp_old - logp), 2)) / self.n_traj
        else:
            mean_kl = (1 - self.gamma) * torch.sum(
                self.hratio[inds] * self.gamma_discount * (logp_old - logp)) / self.n_traj

        return mean_kl

def run_htrpo_train(env, agent, max_timesteps, logger, eval_interval = None, num_evals = 5):
    timestep_counter = 0
    total_updates = max_timesteps // agent.nsteps
    epinfobuf = deque(maxlen=100)
    success_history = deque(maxlen=100)
    ep_num = 0

    if eval_interval:
        # eval_ret, eval_suc = agent.eval_brain(env, render=False, eval_num=1000)
        eval_ret = agent.eval_brain(env, render=False, eval_num=50)
        print("evaluation_eprew:".ljust(20) + str(np.mean(eval_ret)))
        # print("evaluation_sucrate:".ljust(20) + str(np.mean(eval_suc)))
        logger.add_scalar("episode_reward/train", np.mean(eval_ret), timestep_counter)

    while (True):
        mb_obs, mb_rewards, mb_actions, mb_dones, mb_logpacs, mb_obs_, mb_mus, mb_sigmas \
            , mb_distris = [], [], [], [], [], [], [], [], []
        mb_dg, mb_ag = [], []
        epinfos = []
        successes = []
        obs_dict = env.reset()
        # env.render()

        for i in range(0, agent.nsteps, env.num_envs):
            for key in obs_dict.keys():
                obs_dict[key] = torch.Tensor(obs_dict[key])
            # observations = torch.cat((obs_dict['observation'], obs_dict['desired_goal'] - obs_dict['achieved_goal']),
            #                          dim=-1)

            observations = torch.cat((obs_dict['observation'], obs_dict['desired_goal']), dim = -1)
            if not agent.dicrete_action:
                actions, mus, logsigmas, sigmas = agent.choose_action(observations)
                logp = agent.compute_logp(mus, logsigmas, sigmas, actions)
                mus = mus.cpu().numpy()
                sigmas = sigmas.cpu().numpy()
                mb_mus.append(mus)
                mb_sigmas.append(sigmas)
            else:
                actions, distris = agent.choose_action(observations)
                logp = agent.compute_logp(distris, actions)
                distris = distris.cpu().numpy()
                mb_distris.append(distris)
            observations = obs_dict['observation'].cpu().numpy()
            actions = actions.cpu().numpy()
            logp = logp.cpu().numpy()
            obs_dict_, rewards, dones, infos = env.step(actions)

            # if timestep_counter > 350000:
            # env.render()

            mb_obs.append(observations)
            mb_actions.append(actions)
            mb_logpacs.append(logp)
            mb_dones.append(dones.astype(np.uint8))
            mb_rewards.append(rewards)
            mb_obs_.append(obs_dict_['observation'].copy())
            mb_dg.append(obs_dict_['desired_goal'].copy())
            mb_ag.append(obs_dict_['achieved_goal'].copy())

            for e, info in enumerate(infos):
                if dones[e]:
                    epinfos.append(info.get('episode'))
                    successes.append(info.get('is_success'))
                    for k in obs_dict_.keys():
                        obs_dict_[k][e] = info.get('new_obs')[k]
                    ep_num += 1

            obs_dict = obs_dict_

        epinfobuf.extend(epinfos)
        success_history.extend(successes)

        # make all final states marked by done, preventing wrong estimating of returns and advantages.
        # done flag:
        #      0: undone and not the final state
        #      1: realdone
        #      2: undone but the final state
        ep_num += (mb_dones[-1] == 0).sum()
        mb_dones[-1][np.where(mb_dones[-1] == 0)] = 2

        def reshape_data(arr):
            s = arr.shape
            return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

        mb_obs = reshape_data(np.asarray(mb_obs, dtype=np.float32))
        mb_rewards = reshape_data(np.asarray(mb_rewards, dtype=np.float32))
        mb_actions = reshape_data(np.asarray(mb_actions))
        mb_logpacs = reshape_data(np.asarray(mb_logpacs, dtype=np.float32))
        mb_dones = reshape_data(np.asarray(mb_dones, dtype=np.uint8))
        mb_obs_ = reshape_data(np.asarray(mb_obs_, dtype=np.float32))
        mb_ag = reshape_data(np.asarray(mb_ag, dtype=np.float32))
        mb_dg = reshape_data(np.asarray(mb_dg, dtype=np.float32))

        assert mb_obs.ndim <= 2 and mb_rewards.ndim <= 2 and mb_actions.ndim <= 2 and \
               mb_logpacs.ndim <= 2 and mb_dones.ndim <= 2 and mb_obs_.ndim <= 2, \
            "databuffer only supports 1-D data's batch."

        if not agent.dicrete_action:
            mb_mus = reshape_data(np.asarray(mb_mus, dtype=np.float32))
            mb_sigmas = reshape_data(np.asarray(mb_sigmas, dtype=np.float32))
            assert mb_mus.ndim <= 2 and mb_sigmas.ndim <= 2, "databuffer only supports 1-D data's batch."
        else:
            mb_distris = reshape_data(np.asarray(mb_distris, dtype=np.float32))
            assert mb_distris.ndim <= 2, "databuffer only supports 1-D data's batch."

        # store transition
        transition = {
            'state': mb_obs if mb_obs.ndim == 2 else np.expand_dims(mb_obs, 1),
            'action': mb_actions if mb_actions.ndim == 2 else np.expand_dims(mb_actions, 1),
            'reward': mb_rewards if mb_rewards.ndim == 2 else np.expand_dims(mb_rewards, 1),
            'next_state': mb_obs_ if mb_obs_.ndim == 2 else np.expand_dims(mb_obs_, 1),
            'done': mb_dones if mb_dones.ndim == 2 else np.expand_dims(mb_dones, 1),
            'logpac': mb_logpacs if mb_logpacs.ndim == 2 else np.expand_dims(mb_logpacs, 1),
            'other_data': {
                'desired_goal': mb_dg if mb_dg.ndim == 2 else np.expand_dims(mb_dg, 1),
                'achieved_goal': mb_ag if mb_ag.ndim == 2 else np.expand_dims(mb_ag, 1),
            }
        }
        if not agent.dicrete_action:
            transition['mu'] = mb_mus if mb_mus.ndim == 2 else np.expand_dims(mb_mus, 1)
            transition['sigma'] = mb_sigmas if mb_sigmas.ndim == 2 else np.expand_dims(mb_sigmas, 1)
        else:
            transition['distri'] = mb_distris if mb_distris.ndim == 2 else np.expand_dims(mb_distris, 1)
        agent.store_transition(transition)

        # agent learning step
        agent.learn()

        # training controller
        timestep_counter += agent.nsteps
        if timestep_counter > max_timesteps:
            break

        print("------------------log information------------------")
        print("total_timesteps:".ljust(20) + str(timestep_counter))
        print("iterations:".ljust(20) + str(agent.learn_step_counter) + " / " + str(int(total_updates)))
        if agent.value_type is not None:
            explained_var = explained_variance(agent.V.cpu().numpy(), agent.esti_R.cpu().numpy())
            print("explained_var:".ljust(20) + str(explained_var))
            logger.add_scalar("explained_var/train", explained_var, timestep_counter)
        print("episode_len:".ljust(20) + "{:.1f}".format(np.mean([epinfo['l'] for epinfo in epinfobuf])))
        rew = np.mean([epinfo['r'] for epinfo in epinfobuf]) + agent.max_steps
        print("episode_rew:".ljust(20) + str(rew))
        logger.add_scalar("episode_reward/train", rew, timestep_counter)
        print("success_rate:".ljust(20) + "{:.3f}".format(100 * np.mean(success_history)) + "%")
        logger.add_scalar("success_rate/train", np.mean(success_history), timestep_counter)
        print("mean_kl:".ljust(20) + str(agent.cur_kl))
        logger.add_scalar("mean_kl/train", agent.cur_kl, timestep_counter)
        print("policy_ent:".ljust(20) + str(agent.policy_ent))
        logger.add_scalar("policy_ent/train", agent.policy_ent, timestep_counter)
        print("value_loss:".ljust(20) + str(agent.value_loss))
        logger.add_scalar("value_loss/train", agent.value_loss, timestep_counter)
        print("actual_imprv:".ljust(20) + "{:.5f}".format(agent.improvement))
        logger.add_scalar("actual_imprv/train", agent.improvement, timestep_counter)
        print("exp_imprv:".ljust(20) + "{:.5f}".format(agent.expected_improvement))
        logger.add_scalar("exp_imprv/train", agent.expected_improvement, timestep_counter)
        print("valid_ep_ratio:".ljust(20) + "{:.3f}".format(agent.n_valid_ep / ep_num))
        logger.add_scalar("valid_ep_ratio/train", agent.n_valid_ep / ep_num, timestep_counter)

        ep_num = 0

        if eval_interval and timestep_counter % eval_interval == 0:
            agent.save_model("output/models/HTRPO")
            eval_ret = agent.eval_brain(env, render=False, eval_num=num_evals)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("evaluation_eprew:".ljust(20) + str(np.mean(eval_ret)))
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logger.add_scalar("episode_reward/eval", np.mean(eval_ret), timestep_counter)

    return agent
