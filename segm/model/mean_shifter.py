import torch
import torch.nn as nn

class MeanShifter(nn.Module):
    def __init__(self, d_encoder, d_meanshift, n_layers):
        super().__init__()
        '''
        self.blocks = nn.ModuleList(
            [Similarity(d_encoder, d_meanshift) for i in range(n_layers)]
        )
        '''
        self.n_layers = n_layers
        self.scale = d_meanshift ** -0.5
        self.norm = nn.LayerNorm(d_encoder)
        self.q = nn.Linear(d_encoder, d_meanshift)

    def forward(self, x):
        '''
        for blk in self.blocks:
            x = blk(x)
        '''
        for i in range(self.n_layers):
            q = self.q(x)
            similarity = (q @ q.transpose(-2, -1)) * self.scale
            similarity = similarity.softmax(dim=-1)
            x = self.norm(similarity @ x)
        return x

'''
class Similarity(nn.Module):
    def __init__(self, d_token, d_meanshift):
        super().__init__()
        self.scale = d_meanshift ** -0.5
        self.q = nn.Linear(d_token, d_meanshift)
        self.norm = nn.LayerNorm(d_encoder)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        q = self.q(x)
        similarity = (q @ q.transpose(-2, -1)) * self.scale
        similarity = similarity.softmax(dim=-1)
        y = similarity @ x
        y = self.norm(y)
        return y, similarity
'''