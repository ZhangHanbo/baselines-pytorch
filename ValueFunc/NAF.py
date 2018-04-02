import torch
from Feature_Extractor.MLP import MLP
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

class FCNAF(MLP):
    def __init__(self,
                 n_inputfeats,    # input dim
                 n_actions,   # action dim
                 n_hiddens = [30],  # hidden unit number list
                 nonlinear = F.relu,
                 action_active = None,
                 action_scaler = None
                 ):
        self.n_actions = n_actions
        n_outputfeats = 1 + self.n_actions + self.n_actions * self.n_actions
        super(FCNAF, self).__init__(n_inputfeats, n_outputfeats, n_hiddens,
                                    nonlinear)
        # these two lines cant be moved.
        self.action_active = action_active
        self.action_scaler = action_scaler
        if action_scaler is not None:
            if isinstance(action_scaler, (int, long, float)):
                self.action_scaler = Variable(torch.Tensor([action_scaler]))
            else:
                self.action_scaler = Variable(torch.Tensor(action_scaler))

    def forward(self,x):
        x = MLP.forward(self, x)
        # x is a batch
        if x.dim() == 2:
            mu = x[:,:self.n_actions]
            if self.action_active is not None:
                if self.action_scaler is not None:
                    mu = self.action_scaler * self.action_active(mu)
                else:
                    mu = self.action_active(mu)
            V = x[:,self.n_actions:self.n_actions+1]
            L_vectors = []
            mask_vectors = []
            for t in range(self.n_actions):
                L_vectors.append( x[:,self.n_actions + 1 + t * self.n_actions : \
                                   self.n_actions + 1 + (t+1) * self.n_actions].unsqueeze(0))
                if t < self.n_actions - 1:
                    mask_vec_z = Variable(torch.zeros(1, self.n_actions - 1 - t))
                    mask_vec_o = Variable(torch.ones(1, 1 + t))
                    mask_vectors.append(torch.cat([mask_vec_z, mask_vec_o], 1))
                else:
                    mask_vectors.append(Variable(torch.ones(1, 1 + t)))
            Lunmasked_ = torch.cat(L_vectors,0)
            mask = torch.cat(mask_vectors,0)
            Lunmasked = Lunmasked_.permute([1,0,2])
            L = torch.mul(Lunmasked, mask)
        elif x.dim() == 1:
            mu = x[:self.n_actions]
            if self.action_active is not None:
                if self.action_scaler is not None:
                    mu = self.action_scaler * self.action_active(mu)
                else:
                    mu = self.action_active(mu)
            V = x[self.n_actions]
            L_vectors = []
            mask_vectors = []
            for t in range(self.n_actions):
                L_vectors.append(x[self.n_actions + 1 + t * self.n_actions : \
                                self.n_actions + 1 + (t+1) * self.n_actions].unsqueeze(0))
                if t < self.n_actions - 1:
                    mask_vec_z = Variable(torch.zeros(1, self.n_actions - 1 - t))
                    mask_vec_o = Variable(torch.ones(1, 1 + t))
                    mask_vectors.append(torch.cat([mask_vec_z, mask_vec_o], 1))
                else:
                    mask_vectors.append(Variable(torch.ones(1, 1 + t)))
            Lunmasked = torch.cat(L_vectors,0)
            mask = torch.cat(mask_vectors,0)
            L = torch.mul(Lunmasked,mask)
        else:
            raise RuntimeError("dimenssion not matched")
        return V,mu,L
