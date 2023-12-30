import torch
from torch import nn
from .embedding import Temporal
from .translayer import Transformer

class Enhancer(nn.Module):
    def __init__(self, in_channel=1024, dropout=0):
        super(Enhancer, self).__init__()
        self.embedding = Temporal(in_channel, 512)
        self.transformer = Transformer(512, 2, 4, 128, 512, dropout)
    
    def forward(self, x):
        # x: (B, T, C)
        x = self.embedding(x)
        x = self.transformer(x)
        return x
    
if __name__ == '__main__':
    enhancer = Enhancer(1024, 0).cuda()
    x = torch.randn(5, 10, 1024).cuda()
    y = enhancer(x)
    # 5 x 10 x 512
    print(y.shape)