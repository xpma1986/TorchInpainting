import torch
from torch.optim import Adam
from model.unet import UNet
from data.image_net import ImageNetDataGenerator
from model.losses import ContentLoss

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    max_epoch = 1000
    epoch0 = 0

    train_data = ImageNetDataGenerator()
    eval_data = ImageNetDataGenerator(dir='images/image_net/ILSVRC2012_img_val')

    model = UNet(shape_in=[3,512,512], shape_out=[3,512,512])
    optimizer = Adam(model.parameters(), lr=0.00001)
    criterion = ContentLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loss = []
    eval_loss = []

    for epoch in range(epoch0, max_epoch):
        print('== Epoch ' + str(epoch) + ' ==')

        model.train()
        cur_train_loss = 0

        for batch in range(len(train_data)):
            optimizer.zero_grad()
            
            image, mask, masked = train_data[batch]

            

