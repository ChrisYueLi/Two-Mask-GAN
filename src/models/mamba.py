import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

# Helper functions
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

# Define Mamba components based on the provided architecture

class Mamba(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.conv = nn.Conv1d(dim, inner_dim, 1)
        self.ssm = SSM(dim, inner_dim)  # Selective State Space Model
        self.ff = FeedForward(dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        # Integrate Conv, SSM, and FFN modules
        out = self.conv(out.transpose(1, 2)).transpose(1, 2)
        out = self.ssm(out)
        out = self.ff(out)

        return self.norm(out)

class SSM(nn.Module):
    def __init__(self, dim, inner_dim):
        super().__init__()
        self.A = nn.Linear(dim, inner_dim)
        self.B = nn.Linear(dim, inner_dim)
        self.C = nn.Linear(inner_dim, dim)
        self.D = nn.Linear(inner_dim, dim)

    def forward(self, x):
        h = torch.tanh(self.A(x) + self.B(x))
        h = torch.relu(self.C(h) + self.D(h))
        return h

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
