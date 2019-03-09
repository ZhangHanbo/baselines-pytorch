import abc
import copy
import numpy as np
from config import DATABUFFER_CONFIG

class databuffer(object):
    def __init__(self, hyperparams):
        config = copy.deepcopy(DATABUFFER_CONFIG)
        config.update(hyperparams)
        self.max_size = config['memory_size']
        self.state_dims = config['n_states']
        self.actions_dims = config['n_actions']

        self.S = np.zeros([])
        self.A = np.zeros([])
        self.R = np.zeros([])
        self.done = np.zeros([])

        # memory counter: How many transitions are recorded in total
        self.mem_c = 0

    def store_transitions(self, transitions):
        self.S = np.concatenate((self.S, transitions['state']), axis=0)
        self.A = np.concatenate((self.A, transitions['action']), axis=0)
        self.R = np.concatenate((self.R, transitions['reward']), axis=0)
        self.done = np.concatenate((self.done, transitions['done']), axis=0)
        self.mem_c += self.S.shape[0]
        if self.mem_c >self.max_size:
            self.S = self.S[-self.max_size:]
            self.A = self.A[-self.max_size:]
            self.R = self.R[-self.max_size:]
            self.done = self.done[-self.max_size:]

    def sample_batch(self, batch_size):
        sample_index = np.random.choice(min(self.max_size, self.mem_c), size=batch_size)
        batch = {}
        batch['state'] = self.S[sample_index]
        batch['action'] = self.A[sample_index]
        batch['reward'] = self.R[sample_index]
        batch['done'] = self.done[sample_index]
        return batch, sample_index

    def reset_buffer(self):
        self.S = np.zeros([])
        self.A = np.zeros([])
        self.R = np.zeros([])
        self.done = np.zeros([])
        self.mem_c = 0

class databuffer_PG_gaussian(databuffer):
    def __init__(self, hyperparams):
        super(databuffer_PG_gaussian, self).__init__(hyperparams)
        self.mu = np.zeros([])
        self.sigma = np.zeros([])

    def store_transitions(self, transitions):
        databuffer.store_transitions(self, transitions)
        self.mu = np.concatenate((self.mu, transitions['mu']), axis=0)
        self.sigma = np.concatenate((self.sigma, transitions['sigma']), axis=0)
        if self.mem_c >self.max_size:
            self.mu = self.mu[-self.max_size:]
            self.sigma = self.sigma[-self.max_size:]

    def sample_batch(self, batch_size):
        batch, sample_index = databuffer.sample_batch(self, batch_size)
        batch['mu'] = self.mu[sample_index]
        batch['sigma'] = self.sigma[sample_index]
        return batch, sample_index

    def reset_buffer(self):
        databuffer.reset_buffer(self)
        self.mu = np.zeros([])
        self.sigma = np.zeros([])


