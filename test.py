from network import MLP
import torch
from torch.autograd import Variable

'''
a = MLP(n_inputfeats=100, n_outputfeats=10, n_hiddens = [50,50,50])
input = Variable(torch.rand(3,100))
output = a(input)
print(output)
meanout = output.mean()
meanout.backward()
'''

