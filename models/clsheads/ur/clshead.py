import torch
import torch.nn as nn

class URClsHead(nn.Module):
    def __init__(self, in_channel=1024):
        super(URClsHead, self).__init__()
        self.in_channel = in_channel
        self.build_layers()
    
    def build_layers(self):
        ratio1 = 8
        self.fc1 = nn.Linear(self.in_channel, self.in_channel // ratio1)
        self.fc2 = nn.Linear(self.in_channel // ratio1, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (BN, T, C)
        x = self.relu(self.fc1(x))
        score = self.sigmoid(self.fc2(x))

        return score

if __name__ == '__main__':
    clshead = URClsHead(1024, 0.7).cuda()
    x = torch.randn(5, 10, 1024).cuda()
    score = clshead(x)
    print(score.shape)
