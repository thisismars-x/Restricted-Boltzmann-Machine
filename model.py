import torch
from torch.nn import functional as f


from torch import nn
class rboltzmann(nn.Module):

    '''
    visible:    SIZE OF VISIBLE LAYER
    hidden :    SIZE OF HIDDEN LAYER
    sample :    GIBBS SAMPLING RATE
    '''

    def __init__(self, visible = 784, hidden = 128, sample = 1):
        super().__init__()

        self.v, self.h = nn.Parameter(torch.randn(1, visible)), nn.Parameter(torch.randn(1, hidden))
        self.w = nn.Parameter(torch.randn(hidden, visible))
        self.sample = sample

    def to_hidden(self, v):

        '''
        sample conditionally a hidden variable for a visible one
        '''

        return torch.sigmoid(f.linear(v, self.w, self.h)).bernoulli()

    def to_visible(self, h):

        '''
        sample conditionally a visible variable for a hidden one
        '''

        return torch.sigmoid(f.linear(h, self.w.t(), self.v)).bernoulli()

    def free_energy(self, v):

        v_term = v @ self.v.t()
        w_x_h = f.linear(v, self.w, self.h)
        h_term = torch.sum(f.softplus(w_x_h), dim = 1)

        return torch.mean(-h_term -v_term)

    def forward(self, v):

        h = self.to_hidden(v)
        for i in range(self.sample):
            gibb = self.to_visible(h)
            h = self.to_hidden(gibb)

        return v, gibb

