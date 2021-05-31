from torch.nn import Module, Conv2d

from model.encoder import Encoder
from model.decoder import Decoder

class UNet(Module):
    def __init__(self, shape_in=[3, 512, 512], shape_out=[3, 512, 512]):
        super(UNet, self).__init__()

        self.shape_in = shape_in
        self.shape_out = shape_out

        self.encoder = Encoder([3, 16, 32, 64, 128, 256, 512])
        self.decoder = Decoder([512, 256, 128, 64, 32, 16, 8], 
                            [256, 128, 64, 32, 16, 3])
        
        self.conv = Conv2d(8, 3, kernel_size=3, padding=1)

    def forward(self, x):
        e_out = self.encoder(x, return_sequences=True)
        e_out.reverse()

        d_out = self.decoder(e_out[0], concat=e_out[1:])
        out = self.conv(d_out)

        return out
