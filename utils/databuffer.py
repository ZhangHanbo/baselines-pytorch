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
        if 'n_actions' in config.keys():
            self.n_actions = config['n_actions']
        self.actions_dims = config['n_action_dims']

        self.S = np.zeros([0, self.state_dims], dtype = np.float32)
        self.A = np.zeros([0, self.actions_dims], dtype = np.uint8 if config['dicrete_action'] else np.float32)
        self.R = np.zeros([0, 1])
        self.S_ = np.zeros([0, self.state_dims], dtype = np.float32)
        self.done = np.zeros([0, 1])

        # memory counter: How many transitions are recorded in total
        self.mem_c = 0

    def store_transition(self, transitions):
        self.S = np.concatenate((self.S, transitions['state']), axis=0)
        self.A = np.concatenate((self.A, transitions['action']), axis=0)
        self.R = np.concatenate((self.R, transitions['reward']), axis=0)
        self.done = np.concatenate((self.done, transitions['done']), axis=0)
        self.S_ = np.concatenate((self.S_, transitions['next_state']), axis=0)
        self.mem_c += transitions['state'].shape[0]
        if self.mem_c >self.max_size:
            self.S = self.S[-self.max_size:]
            self.A = self.A[-self.max_size:]
            self.R = self.R[-self.max_size:]
            self.done = self.done[-self.max_size:]

    def sample_batch(self, batch_size = None):
        if batch_size is not None:
            if batch_size > self.mem_c or batch_size > self.max_size:
                raise RuntimeError("Batch size is bigger than buffer size")
            sample_index = np.random.choice(min(self.max_size, self.mem_c), size=batch_size)
        else:
            sample_index = np.arange(min(self.max_size, self.mem_c))
        batch = {}
        batch['state'] = self.S[sample_index]
        batch['action'] = self.A[sample_index]
        batch['reward'] = self.R[sample_index]
        batch['done'] = self.done[sample_index]
        batch['next_state'] = self.S_[sample_index]
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
        self.mu = np.zeros([0, self.actions_dims])
        self.sigma = np.zeros([0, self.actions_dims])

    def store_transition(self, transitions):
        databuffer.store_transition(self, transitions)
        self.mu = np.concatenate((self.mu, transitions['mu']), axis=0)
        self.sigma = np.concatenate((self.sigma, transitions['sigma']), axis=0)
        if self.mem_c >self.max_size:
            self.mu = self.mu[-self.max_size:]
            self.sigma = self.sigma[-self.max_size:]

    def sample_batch(self, batch_size= None):
        batch, sample_index = databuffer.sample_batch(self, batch_size)
        batch['mu'] = self.mu[sample_index]
        batch['sigma'] = self.sigma[sample_index]
        return batch, sample_index

    def reset_buffer(self):
        databuffer.reset_buffer(self)
        self.mu = np.zeros([])
        self.sigma = np.zeros([])

class databuffer_PG_softmax(databuffer):
    def __init__(self, hyperparams):
        super(databuffer_PG_softmax, self).__init__(hyperparams)
        self.distr = np.zeros([0, self.n_actions])

    def store_transition(self, transitions):
        databuffer.store_transition(self, transitions)
        self.distr = np.concatenate((self.distr, transitions['distr']), axis=0)
        if self.mem_c >self.max_size:
            self.distr = self.distr[-self.max_size:]

    def sample_batch(self, batch_size= None):
        batch, sample_index = databuffer.sample_batch(self, batch_size)
        batch['distr'] = self.distr[sample_index]
        return batch, sample_index

    def reset_buffer(self):
        databuffer.reset_buffer(self)
        self.distr = np.zeros([])

