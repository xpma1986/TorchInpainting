from torch.nn import Module, ModuleList
from torchvision.models import vgg16

class VGG16(Module):
    def __init__(self, slices=[6, 11, 19], slice_start=2):
        self.vgg16 = vgg16(pretrained=True)
        self.slices = slices

        vgg_modules = []
        for n in range(len(slices)):
            modules = []
            for i, m in enumerate(self.vgg16.modules()):
                if i >= slice_start and i <= slices[n]:
                    modules.append(m)

            vgg_modules.append(ModuleList(modules))

        self.vgg = ModuleList(vgg_modules)

    def forward(self, x):
        out = []

        for _, vgg in enumerate(self.vgg):
            out.append(vgg(x))

        return out
