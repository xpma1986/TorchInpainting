from torch.nn.modules.container import ModuleList
from model.block import Block
from torch.nn import Module

class Encoder(Module):
    def __init__(self, filters):
        super(Encoder, self).__init__()

        self.n_blks = len(filters) - 1
        self.filters = filters
        
        blk_list = []

        for n in range(self.n_blks):
            blk_list.append(Block([filters[n], filters[n+1]], stride=2))

        self.blk_list = ModuleList(blk_list)

    def forward(self, x, return_sequences=False):
        if return_sequences:
            out = [x]

            for _, blk in enumerate(self.blk_list):
                out.append(blk(out[-1]))

            return out
        else:
            out = x
            for _, m in enumerate(self.blk_list):
                out = m(out)

            return out


