import torch
from torch.nn import *

class Block(Module):
    def __init__(self, filters=[], kernel_size=3, stride=1, batch_norm=True, scale_factor=1, activation=ReLU()):
        super(Block, self).__init__()
        
        self.filters = filters

        self.scale_factor = scale_factor
        if scale_factor > 1:
            self.up = Upsample(scale_factor=scale_factor)

        self.conv = Conv2d(filters[0], filters[1], kernel_size=kernel_size, stride=stride, padding=1)

        if batch_norm:
            self.bn = BatchNorm2d(filters[1])
        else:
            self.bn = None

        self.activation = activation

    def forward(self, x, concat=None):
        out = self.up(x) if self.scale_factor > 1 else x

        if concat != None:
            out = torch.cat([out, concat], dim=1)

        out = self.conv(out)
        
        if self.bn != None:
            out = self.bn(out)

        return self.activation(out) if self.activation != None else out
