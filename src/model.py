import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.15):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.n1 = nn.InstanceNorm2d(out_ch, affine=True)
        self.n2 = nn.InstanceNorm2d(out_ch, affine=True)
        self.d  = nn.Dropout2d(p_drop)
    def forward(self, x):
        x = F.relu(self.n1(self.c1(x)))
        x = self.d(x)
        x = F.relu(self.n2(self.c2(x)))
        return x

class UNetTiny(nn.Module):
    def __init__(self, base=8, p_drop=0.15):
        super().__init__()
        self.e1 = ConvBlock(1, base, p_drop)
        self.p1 = nn.MaxPool2d(2)
        self.e2 = ConvBlock(base, base*2, p_drop)
        self.p2 = nn.MaxPool2d(2)
        self.mid= ConvBlock(base*2, base*4, p_drop)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.d2 = ConvBlock(base*4, base*2, p_drop)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.d1 = ConvBlock(base*2, base, p_drop)
        self.out= nn.Conv2d(base, 1, 1)
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        m  = self.mid(self.p2(e2))
        d2 = self.u2(m)
        d2 = self.d2(torch.cat([d2, e2], dim=1))
        d1 = self.u1(d2)
        d1 = self.d1(torch.cat([d1, e1], dim=1))
        r  = torch.tanh(self.out(d1)) * 0.20
        return torch.clamp(x + r, 0.0, 1.0)
