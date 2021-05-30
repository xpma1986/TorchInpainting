import os
import torch
from torch.optim import Adam

from model.unet import UNet
from model.losses import ContentLoss
from data.image_net import ImageNetDataGenerator

import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    work_dir = 'outputs/unet'

    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    max_epoch = 1000
    epoch0 = 0

    train_data = ImageNetDataGenerator()
    eval_data = ImageNetDataGenerator(dir='images/image_net/ILSVRC2012_img_val')

    model = UNet(shape_in=[3,512,512], shape_out=[3,512,512])
    optimizer = Adam(model.parameters(), lr=0.00001)
    criterion = ContentLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loss = []
    eval_loss = []
    accuracy = []

    for epoch in range(epoch0, max_epoch):
        print('== Epoch ' + str(epoch) + ' ==')

        model.train()
        cur_train_loss = 0

        for batch in range(len(train_data)):
            optimizer.zero_grad()

            image, mask, masked = train_data[batch]
            image_tensor = torch.as_tensor(image, device=device)
            mask_tensor = torch.as_tensor(mask, device=device)
            masked_tensor = torch.as_tensor(masked, device=device)

            inpainted_tensor = model(masked_tensor)

            loss = criterion(image_tensor, mask_tensor, inpainted_tensor)
            loss.backward()
            optimizer.step()

            cur_train_loss += loss.item()
            print('[' + str(batch+1) + '/' +  str(len(train_data)) + '] Train Loss: ' + str(cur_train_loss/float(batch+1)), end='\r')

        cur_train_loss /= len(train_data)
        train_loss.append(cur_train_loss)
        print('Train Loss: ' + str(train_loss[-1]))

        model.eval()
        cur_eval_loss = 0

        with torch.no_grad():
            for batch in range(len(eval_data)):
                image, mask, masked = eval_data[batch]
                image_tensor = torch.as_tensor(image, device=device)
                mask_tensor = torch.as_tensor(mask, device=device)
                masked_tensor = torch.as_tensor(masked, device=device)

                inpainted_tensor = model(masked_tensor)
                loss = criterion(image_tensor, mask_tensor, inpainted_tensor)

                cur_eval_loss += loss.item()
                print('[' + str(batch+1) + '/' +  str(len(eval_data)) + '] Eval Loss: ' + str(cur_eval_loss/float(batch+1)), end='\r')

                if batch == 0:
                    path = work_dir + '/' + str(epoch)

                    if not os.path.exists(path):
                        os.mkdir(path)
                    
                    mask = np.transpose(mask, [0,2,3,1])
                    image = np.transpose(image, [0,2,3,1])
                    masked = np.transpose(masked, [0,2,3,1])
                    inpainted = inpainted_tensor.view()
                    inpainted = np.transpose(inpainted, [0,2,3,1])

                    for n in range(eval_data.batch_size):
                        image[n,:,:,:] = cv2.cvtColor(image[n,:,:,:], cv2.COLOR_BGR2RGB)
                        masked[n,:,:,:] = cv2.cvtColor(masked[n,:,:,:], cv2.COLOR_BGR2RGB)
                        inpainted[n,:,:,:] = cv2.cvtColor(inpainted[n,:,:,:], cv2.COLOR_BGR2RGB)

                        _, axes = plt.subplots(2, 2, figsize=(10, 10))
                        axes[0, 0].imshow(image[n,:,:,:])
                        axes[0, 1].imshow(masked[n,:,:,:])
                        axes[1, 0].imshow(inpainted[n,:,:,:])
                        axes[1, 1].imshow(mask[n,:,:,0], cmap='gray')
                    
                        axes[0, 0].set_title('Original Image')
                        axes[0, 1].set_title('Masked Image')
                        axes[1, 0].set_title('Inpainted Image')
                        axes[1, 1].set_title('Mask')
                    
                        plt.savefig('{}/{}/{}.png'.format(work_dir, epoch, n))
                        plt.close()

            cur_eval_loss /= len(eval_data)
            eval_loss.append(cur_eval_loss)

        epochs = np.arange(len(train_loss))

        plt.figure(figsize=(5, 5))
        plt.plot(epochs, train_loss, label='Train Loss')
        plt.plot(epochs, eval_loss, label='Eval Loss')
        plt.legend()

        plt.savefig(work_dir + '/loss.png')
        plt.close()

        torch.save(model.state_dict, work_dir + '/' + str(epoch) + '/model.h5')

            

