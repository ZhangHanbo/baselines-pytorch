import numpy as np
from sklearn.neighbors import KernelDensity
import torch
import torch.nn.functional as F
import os
from scipy.special import entr
import pickle
from collections import deque

class KernalDensityEstimator(object):
    def __init__(self, name, logger, samples=10000, kernel='gaussian', bandwidth=0.2, normalize=True):

        self.kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.normalize = normalize
        self.mean = 0.
        self.std = 1.
        self.fitted_kde = None
        self.name = name
        self.logger = logger
        self.buffer = None
        self.w_controller = 0
        self.n_data = 0
        self.time_steps = 0
        self.n_kde_samples = samples
        self.kde_samples = None

    def fit(self, n_kde_samples=10000):
        self.kde_samples = self._sample(n_kde_samples)
        if self.normalize:
            self.mean = np.mean(self.kde_samples, axis=0, keepdims=True)
            self.std = np.std(self.kde_samples, axis=0, keepdims=True) + 1e-4
            self.kde_samples = (self.kde_samples - self.mean) / self.std

        self.fitted_kde = self.kde.fit(self.kde_samples)

        # Scoring samples is a bit expensive, so just use 1000 points
        num_samples = 1000
        s = self.fitted_kde.sample(num_samples)
        entropy = - self.fitted_kde.score(s) / num_samples + np.log(self.kde_sample_std).sum()
        self.logger.add_scalar('{}_entropy'.format(self.module_name), entropy, self.time_steps)

    def normalize_samples(self, samples):
        assert self.normalize
        return (samples - self.mean) / self.std

    def evaluate_log_density(self, samples):
        assert self.fitted_kde is not None
        if self.normalize:
            samples = self.normalize_samples(samples)
        return self.fitted_kde.score_samples(samples)

    def evaluate_elementwise_entropy(self, samples, beta=0.):
        if self.normalize:
            samples = self.normalize_samples(samples)
        log_px = self.fitted_kde.score_samples(samples)
        px = np.exp(log_px)
        elem_entropy = entr(px + beta)
        return elem_entropy

    def save(self, save_folder):
        with open(os.path.join(save_folder, self.name + "_density_estimator.pkl")) as f:
            pickle.dump(self, f)

    def load(self, save_folder):
        with open(os.path.join(save_folder, self.name + "_density_estimator.pkl")) as f:
            loaded = pickle.load(f)
        for k, v in loaded.__dict__.items():
            self.__dict__[k] = v

    def extend(self, data):
        data = np.asarray(data)
        assert len(data.shape) == 2
        if self.buffer is not None:
            batch_size = data.shape[0]
            buffer_size = self.buffer.shape[0]
            data_dim = self.buffer.shape[1]
            assert data.shape[-1] == data_dim
            if self.w_controller + batch_size > buffer_size:
                # reset data number
                self.w_controller = batch_size + self.w_controller - buffer_size
                self.buffer[self.w_controller - batch_size:] = data[:batch_size - self.w_controller]
                self.buffer[:self.w_controller] = data[- self.w_controller:]
            else:
                self.buffer[self.w_controller:self.w_controller + batch_size] = data
                self.w_controller += batch_size
            self.n_data = np.clip(self.n_data + batch_size, a_min=0, a_max=buffer_size)
            self.time_steps += batch_size
        else:
            # initialize buffer according to the data shape
            self.buffer = np.zeros((1000000, data.shape[1]))
            self.extend(data)

    def _sample(self, batch_size):
        idx = np.random.randint(self.n_data, size=batch_size)
        return self.buffer[idx]


class CuriosityAlphaMixture(object):
    def __init__(self, ag_kde, dg_kde, logger):
        self._alpha = 0.
        self._beta = -3.
        self.ag_kde = ag_kde
        self.dg_kde = dg_kde
        self.logger = logger

    @property
    def alpha(self):
        return self._alpha

    @property
    def time_steps(self):
        return self.dg_kde.time_steps

    def update(self):
        kde_samples = self.dg_kde.kde_samples
        log_p_dg = self.dg_kde.fitted_kde.score_samples(kde_samples)
        log_p_ag = self.ag_kde.fitted_kde.score_samples(kde_samples)
        self._alpha = 1. / max((self._beta + np.mean(log_p_dg) - np.mean(log_p_ag)), 1.)
        self.logger.add_scalar('curiosity_alpha', self._alpha, self.time_steps)
