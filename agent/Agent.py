# father class for all agent
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
import network
import abc

class Agent:
    __metaclass__ = abc.ABCMeta
    def __init__(self,hyperparams):
        pass

    @abc.abstractmethod
    def choose_action(self, s):
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def learn(self):
        raise NotImplementedError("Must be implemented in subclass.")
