from torchvision.models import vgg16

vgg = vgg16(pretrained=False)
for n, m in enumerate(vgg.modules()):
    print(n, '->', m)