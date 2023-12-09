import torch
import torch.nn as nn

class TA(nn.Module):
    def __init__(self,  T,ratio=2):

        super(TA, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv3d(T, T // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(T // ratio, T, 1,bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.avg_pool(x)
        # B,T,C
        out1 = self.sharedMLP(avg)
        max = self.max_pool(x)
        # B,T,C
        out2 = self.sharedMLP(max)
        out = out1+out2

        return out

# task classifictaion or generation
class SCA(nn.Module):
    def __init__(self, in_planes, kerenel_size,ratio = 1):
        super(SCA, self).__init__()
        self.sharedMLP = nn.Sequential(
                nn.Conv2d(in_planes, in_planes // ratio, kerenel_size, padding='same', bias=False),
                nn.ReLU(),
                nn.Conv2d(in_planes // ratio, in_planes, kerenel_size, padding='same', bias=False),)
    def forward(self, x):
        b,t, c, h, w = x.shape
        x = x.flatten(0,1)
        x = self.sharedMLP(x)
        out = x.reshape(b,t, c, h, w)
        return out


