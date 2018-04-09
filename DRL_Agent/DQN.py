import torch
from torch.autograd import Variable
import numpy as np
import Feature_Extractor
from DRL_Agent.Agent import Agent
import copy
from config import DQN_CONFIG
from ApproFunc.DQN import FCDQN

class DQN(Agent):
    def __init__(self, hyperparams):
        config = copy.deepcopy(DQN_CONFIG)
        config.update(hyperparams)
        super(DQN,self).__init__(config)
        self.epsilon_max = config['e_greedy']
        self.replace_target_iter = config['replace_target_iter']
        self.epsilon_increment = config['e_greedy_increment']
        self.epsilon = 0 if self.epsilon_increment is not None else self.epsilon_max
        # initialize zero memory [s, a, r, s_]
        self.memory = torch.Tensor(np.zeros((self.memory_size, self.n_features * 2 + 3)))
        ## TODO: include other network architectures
        self.e_DQN = FCDQN(self.n_features,self.n_actions)
        self.t_DQN = FCDQN(self.n_features,self.n_actions)
        self.lossfunc = config['loss']()
        self.optimizer = config['optimizer'](self.e_DQN.parameters(),lr = self.lr, momentum = self.mom)

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        observation = Variable(torch.Tensor(observation))
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.e_DQN(observation)
            (_ , action) = torch.max(actions_value, 1)
            action = int(action.data[0])
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.hard_update(self.t_DQN, self.e_DQN)
        batch_memory = self.sample_batch()
        r = Variable(batch_memory[:, self.n_features + 1])
        s_ = Variable(batch_memory[:, -self.n_features:])
        q_target = r + self.gamma * torch.max(self.t_DQN(s_),1)[0]
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
