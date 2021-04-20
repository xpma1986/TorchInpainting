import torch
from torch.nn import *

class Block(Module):
    def __init__(self, filters=[], kernel_size=3, stride=1, batch_norm=True, upsampling=False, activation=ReLU()):
        super(Block, self).__init__()
        
        self.filters = filters

        if upsampling:
            self.up = UpsamplingBilinear2d()

        self.conv = Conv2d(filters[0], filters[1], kernel_size=kernel_size, stride=stride, padding=1)

        if batch_norm:
            self.bn = BatchNorm2d(filters[1])

        self.activation = activation

    def forward(self, x, concat=None):
        out = self.up(x) if self.up != None else x

        if concat != None:
            out = torch.cat([out, concat])

        out = self.conv(out)
        
        if self.bn != None:
            out = self.bn(out)

        return self.activation(out) if self.activation != None else out
