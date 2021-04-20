from model.unet import UNet
from data.image_net import ImageNetDataGenerator

max_epoch = 1000
epoch0 = 0

train_data = ImageNetDataGenerator()
eval_data = ImageNetDataGenerator(dir='images/image_net/ILSVRC2012_img_val')

model = UNet(shape_in=[3,512,512], shape_out=[3,512,512])
model.to('cuda')

for epoch in range(epoch0, max_epoch):
    for batch in range(len(train_data)):
        image, mask, masked = train_data[batch]

