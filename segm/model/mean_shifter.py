import torch
import torch.nn as nn

class MeanShifter(nn.Module):
    def __init__(self, d_encoder, d_meanshift, n_layers, use_normalization):
        super().__init__()
        self.n_layers = n_layers
        self.scale = d_meanshift ** -0.5
        self.use_norm = use_normalization
        self.q = nn.Linear(d_encoder, d_meanshift)
        if use_normalization:
            self.norm = nn.LayerNorm(d_encoder)

    def forward(self, x):
        for i in range(self.n_layers):
            q = self.q(x)
            similarity = (q @ q.transpose(-2, -1)) * self.scale
            similarity = similarity.softmax(dim=-1)
            x = similarity @ x
            if self.use_norm:
                x = self.norm(x)
        return x

    def get_attention_map(self, im, layer_id):
        #if layer_id >= self.n_layers or layer_id < 0:
        if not (0 <= layer_id < self.n_layers):
            raise ValueError(f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}.")

        for i in range(layer_id):
            q = self.q(im)
            similarity = (q @ q.transpose(-2, -1)) * self.scale
            similarity = similarity.softmax(dim=-1)
            im = similarity @ im
            if self.use_norm:
                im = self.norm(im)
        q = self.q(im)
        return ((q @ q.transpose(-2, -1)) * self.scale).softmax(dim=-1)