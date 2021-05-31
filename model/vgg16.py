import torch
from torch.nn import Module, ModuleList
from torchvision.models import vgg16

class VGG16(Module):
    def __init__(self, slices=[6, 11, 19], slice_start=2):
        super(VGG16, self).__init__()

        self.vgg16 = vgg16(pretrained=True)
        self.slices = slices
        self.slice_start = slice_start

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg16 = self.vgg16.to(device)

        '''vgg_modules = []
        for n in range(len(slices)):
            modules = []
            for i, m in enumerate(self.vgg16.modules()):
                if i >= slice_start and i <= slices[n]:
                    modules.append(m)

            vgg_modules.append(ModuleList(modules))

        self.vgg = ModuleList(vgg_modules)'''

    def forward(self, x):
        out = []

        slice = 0
        for i, m in enumerate(self.vgg16.modules()):
            if i >= self.slice_start and i <= self.slices[slice]:
                x = m(x)

                if i == self.slices[slice]:
                    out.append(x)
                    slice += 1

                    if slice >= len(self.slices):
                        break

        return out

if __name__ == '__main__':
    vgg = vgg16()

    for i, m in enumerate(vgg.modules()):
        print(str(i) + ':' + str(m))
