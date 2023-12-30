import torch
import torch.nn as nn

class RTFMClsHead(nn.Module):
    def __init__(self, in_channel=512):
        super(RTFMClsHead, self).__init__()
        self.in_channel = in_channel
        self.build_layers()
    
    def build_layers(self):
        ratio1 = 16
        ratio2 = 64
        self.conv1 = nn.Conv1d(self.in_channel, self.in_channel // ratio1, 1, 1, 0)
        self.bn1 = nn.BatchNorm1d(self.in_channel // ratio1)
        
        self.conv2 = nn.Conv1d(self.in_channel // ratio1, self.in_channel // ratio2, 1, 1, 0)
        self.bn2 = nn.BatchNorm1d(self.in_channel // ratio2)

        self.conv3 = nn.Conv1d(self.in_channel // ratio2, 1, 1, 1, 0)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # input: BN x T x C
        # output: BN x 1 x T
        x = self.bn1(self.conv1(x.permute(0, 2, 1)))
        x = self.bn2(self.conv2(self.relu(x)))
        x = self.sigmoid(self.conv3(self.relu(x)))
        return x
    
if __name__ == '__main__':
    clshead = RTFMClsHead(512).cuda()
    x = torch.randn(5, 10, 512).cuda()
    score = clshead(x)
    print(score.shape)