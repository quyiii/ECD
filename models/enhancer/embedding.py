import torch.nn as nn

class Temporal(nn.Module):
    def __init__(self, input_size, out_size):
        super(Temporal, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3,
                    stride=1, padding=1),
            nn.ReLU(),
        )
    def forward(self, x):  
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = x.permute(0, 2, 1)
        return x