import torch
from torch.autograd import Variable
import numpy as np
import basenets
from agents.Agent import Agent
import copy
from config import DQN_CONFIG
from critics.DQN import FCDQN
from utils import databuffer

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
        self.memory = databuffer(config)
        self.batch_size = config['batch_size']
        ## TODO: include other network architectures
        if type(self) == DQN:
            self.e_DQN = FCDQN(self.n_states, self.n_actions, n_hiddens = [50])
            self.t_DQN = FCDQN(self.n_states, self.n_actions, n_hiddens = [50])
            self.lossfunc = config['loss']()
            if self.mom == 0 or self.mom is None:
                self.optimizer = config['optimizer'](self.e_DQN.parameters(),lr = self.lr)
            else:
                self.optimizer = config['optimizer'](self.e_DQN.parameters(), lr=self.lr, momentum = self.mom)

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        observation = torch.Tensor(observation)
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.e_DQN(observation)
            (_ , action) = torch.max(actions_value, 1)
            distri = actions_value.detach().numpy()
            action = int(action.data[0])
        else:
            distri = 1. / self.n_actions * np.ones(self.n_actions)
            action = np.random.randint(0, self.n_actions)
        return action, distri

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.hard_update(self.t_DQN, self.e_DQN)
        batch_memory = self.sample_batch(self.batch_size)[0]

        r = torch.Tensor(batch_memory['reward'])
        s_ = torch.Tensor(batch_memory['next_state'])
        q_target = r + self.gamma * torch.max(self.t_DQN(s_), 1)[0].view(self.batch_size, 1)
        s = torch.Tensor(batch_memory['state'])
        q_eval = self.e_DQN(s)
        q_eval_wrt_a = q_eval.gather(1, torch.LongTensor(batch_memory['action']))
        q_target = q_target.detach()

        self.loss = self.lossfunc(q_eval_wrt_a, q_target)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.cost_his.append(self.loss.data)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
