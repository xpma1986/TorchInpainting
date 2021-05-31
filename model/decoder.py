from torch.nn import Module, ModuleList
from model.block import Block

class Decoder(Module):
    def __init__(self, filters, concat=None):
        super(Decoder, self).__init__()

        self.n_blks = len(filters) - 1
        self.filters = filters

        blk_list = []

        for n in range(self.n_blks):
            filters_in = filters[n] + concat[n] if concat != None else filters[n]
            blk_list.append(Block([filters_in, filters[n+1]], scale_factor=2))

        self.blk_list = ModuleList(blk_list)

    def forward(self, x, concat=None):
        out = x

        for n, blk in enumerate(self.blk_list):
            out = blk(out, concat[n]) if concat != None else blk(out)

        return out

