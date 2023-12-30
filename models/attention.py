import torch
import torch.nn as nn

class ChannelOnlyAttention(nn.Module):

    def __init__(self, in_channels):
        super(ChannelOnlyAttention, self).__init__()

        self.in_channels = in_channels
        self.convq = nn.Conv1d(in_channels=self.in_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.convv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels//2, kernel_size=1, stride=1, padding=0)
        self.convAtten = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels//2, out_channels=self.in_channels//4, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm([self.in_channels//4, 1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.in_channels//4, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_q = self.convq(x) # B x 1 x T
        # print(x_q.shape)
        x_v = self.convv(x) # B x C/2 x T
        # print(x_v.shape)
        x_z = x_v @ x_q.permute(0, 2, 1).softmax(-1) # B x C/2 x 1
        atten = self.convAtten(x_z) # B x C x 1

        return atten

class TemporalOnlyAttention(nn.Module):

    def __init__(self, in_channels):
        super(TemporalOnlyAttention, self).__init__()

        self.in_channels = in_channels
        self.convq = nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels//2, kernel_size=1, stride=1, padding=0)
        self.convv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels//2, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_q = self.convq(x) # B x C/2 x T
        x_q = x_q.mean(-1, keepdim=True) # B x C/2 x 1
        # print(x_q.shape)
        x_v = self.convv(x) # B x C/2 x T
        # print(x_v.shape)
        x_z = x_q.permute(0, 2, 1).softmax(-1) @ x_v # B x 1 x T
        atten = self.sigmoid(x_z) # B x 1 x T

        return atten
