import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
import network
from agent.DQN import DQN
import copy
from config import DQN_CONFIG

class DDQN(DQN):
    def __init__(self,hyperparams):
        config = copy.deepcopy(DQN_CONFIG)
        config.update(hyperparams)
        DQN.__init__(self, config)

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.hard_update(self.t_DQN, self.e_DQN)
        batch_memory = self.sample_batch()

        r = Variable(batch_memory[:, self.n_features + 1])
        s_ = Variable(batch_memory[:, -self.n_features:])
        q_target = self.t_DQN(s_)
        q_eval_wrt_s_ = self.e_DQN(s_)
        a_eval_wrt_s_ = torch.max(q_eval_wrt_s_,1)[1]
        a_indice = [range(q_target.size(0)), list(a_eval_wrt_s_.data.long())]
        q_target = r + self.gamma * q_target[a_indice]

        s = Variable(batch_memory[:, :self.n_features])
        q_eval = self.e_DQN(s)
        a_indice = [range(q_eval.size(0)),list(batch_memory[:, self.n_features].long())]
        q_eval_wrt_a = q_eval[a_indice]
        q_target = q_target.detach()

        self.loss = self.lossfunc(q_eval_wrt_a, q_target)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.cost_his.append(self.loss.data)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
