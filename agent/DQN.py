import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
import network
from agent import Agent
import copy
from config import DQN_CONFIG

class DQN:
    def __init__(self, hyperparams):
        config = copy.deepcopy(DQN_CONFIG)
        config.update(hyperparams)
        self.n_features = config['n_features']
        self.n_actions = config['n_actions']
        self.lr = config['lr']
        self.mom = config['mom']
        self.gamma = config['reward_decay']
        self.epsilon_max = config['e_greedy']
        self.replace_target_iter = config['replace_target_iter']
        self.memory_size = config['memory_size']
        self.batch_size = config['batch_size']
        self.epsilon_increment = config['e_greedy_increment']
        self.epsilon = 0 if self.epsilon_increment is not None else self.epsilon_max
        # total learning step
        self.learn_step_counter = 0
        # initialize zero memory [s, a, r, s_]
        self.memory = torch.Tensor(np.zeros((self.memory_size, self.n_features * 2 + 3)))
        self.cost_his = []
        ## TODO: include other network architectures
        self.e_DQN = network.MLP(self.n_features,self.n_actions)
        self.t_DQN = network.MLP(self.n_features,self.n_actions)
        self.lossfunc = nn.MSELoss()
        self.optimizer = config['optimizer'](self.e_DQN.parameters(),lr = self.lr, momentum = self.mom)

    def store_transition(self, s, a, r, s_, done):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = torch.Tensor(np.hstack((s, a, r, done, s_)))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

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
            self.t_DQN.load_state_dict(self.e_DQN.state_dict())
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        self.r = Variable(batch_memory[:, self.n_features + 1])
        s_ = Variable(batch_memory[:, -self.n_features:])
        q_target = self.r + self.gamma * torch.max(self.t_DQN(s_),1)[0]

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
