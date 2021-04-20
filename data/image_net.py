import cv2
import os
import numpy as np
from copy import deepcopy
from keras.utils import Sequence
from numpy.random import randint

from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

class ImageNetDataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, dir="images/image_net/ILSVRC2012_img_train", batch_size=8, dim=(512, 512), n_channels=3, num_classes=1000, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.num_classes = num_classes
        self.shuffle = shuffle

        self.dir = dir

        self.img_path = []
        self.classes = []
        with open(dir+'/list.txt') as file:
            images = file.read().splitlines()

            for image in images:
                if len(image) > 2:
                    path, class_num = image.split(' ')
                    self.img_path.append(self.dir + '/' + path)
                    self.classes.append(int(class_num))

            self.total = len(self.img_path)
            self.total_images = self.total

            print(self.total)

        self.indices = np.arange(self.total, dtype=np.int)
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.total / self.batch_size))

    def make_mask(self, mask, masked):
        size = int((self.dim[1] + self.dim[0]) * 0.04)
        if self.dim[1] < 64 or self.dim[0] < 64:
            raise Exception("Width and Height of mask must be at least 64!")
        
        # Draw random lines
        for _ in range(randint(2, 20)):
            x1, x2 = randint(1, self.dim[1]), randint(1, self.dim[0])
            y1, y2 = randint(1, self.dim[1]), randint(1, self.dim[0])
            thickness = randint(4, size)
            cv2.line(mask,(x1,y1),(x2,y2),(0,0,0),thickness)
            cv2.line(masked,(x1,y1),(x2,y2),(1,1,1),thickness)
            
        # Draw random circles
        for _ in range(randint(2, 20)):
            x1, y1 = randint(1, self.dim[1]), randint(1, self.dim[0])
            radius = randint(4, size)
            cv2.circle(mask,(x1,y1),radius,(0,0,0), -1)
            cv2.circle(masked,(x1,y1),radius,(1,1,1), -1)
            
        # Draw random ellipses
        for _ in range(randint(2, 20)):
            x1, y1 = randint(1, self.dim[1]), randint(1, self.dim[0])
            s1, s2 = randint(1, self.dim[1]), randint(1, self.dim[0])
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(4, size)
            cv2.ellipse(mask, (x1,y1), (s1,s2), a1, a2, a3, (0,0,0), thickness)
            cv2.ellipse(masked, (x1,y1), (s1,s2), a1, a2, a3, (1,1,1), thickness)
        
    def get_image(self, n, rot_degrees=10):
        image = cv2.imread(self.img_path[n])

        rot = np.random.random()*rot_degrees*2 - rot_degrees
        image, _ = self.rot_degree(image, rot)

        scale = np.random.random()*0.5 + 1.0
        image = cv2.resize(image, (int(scale*image.shape[1]), int(scale*image.shape[0])), interpolation = cv2.INTER_CUBIC)

        height, width = image.shape[:2]
            
        #box = np.zeros(4)
        start_x = np.random.randint(0, width - self.dim[0])
        start_y = np.random.randint(0, height - self.dim[1])
            
        contrast = 1 + np.random.random()*0.2 - 0.1
        brightness = np.random.randint(-10, 10)

        #return image/255.0

        return np.clip((contrast*image[start_y:start_y+self.dim[1], start_x:start_x+self.dim[0], :] + brightness)/255.0, 0, 1)

    
    def __getitem__(self, index, usage='inpainting'):
        'Generate one batch of data'
        # Generate indexes of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        image = np.zeros((self.batch_size, self.n_channels, self.dim[1], self.dim[0]), np.float32)

        if usage == 'classification':
            classes = np.zeros((self.batch_size), dtype=np.long)

            def get_data(n):
                image[n,:,:,:] = np.transpose(self.get_image(indices[n]), (2, 0, 1))
                classes[n] = self.classes[indices[n]]
        elif usage == 'inpainting':
            mask = np.zeros((self.batch_size, self.dim[1], self.dim[0]), np.float32)
            masked = np.zeros((self.batch_size, self.n_channels, self.dim[1], self.dim[0]), np.float32)

            def get_data(n):
                img = self.get_image(indices[n])
                masked_img = np.copy(img)

                self.make_mask(mask[n,:,:], masked_img)

                image[n,:,:,:] = np.transpose(img, [2,0,1])
                masked = np.transpose(masked_img, [2,0,1])
        
        with ThreadPoolExecutor(max_workers=20) as pool:
            all_task = [pool.submit(get_data, n) for n in range(self.batch_size)]
            wait(all_task, return_when=ALL_COMPLETED)
        
        if usage == 'classification':
            return image, classes
        elif usage == 'inpainting':
            return image, mask, masked
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def rot_degree(self, img, degree):
        rows, cols = img.shape[:2]
        center = (cols / 2, rows / 2)
        mask = img.copy()
        mask[:, :] = 255
        M = cv2.getRotationMatrix2D(center, degree, 1)
        top_right = np.array((cols - 1, 0)) - np.array(center)
        bottom_right = np.array((cols - 1, rows - 1)) - np.array(center)
        top_right_after_rot = M[0:2, 0:2].dot(top_right)
        bottom_right_after_rot = M[0:2, 0:2].dot(bottom_right)
        #new_width = max(int(abs(bottom_right_after_rot[0] * 2) + 0.5), int(abs(top_right_after_rot[0] * 2) + 0.5))
        #new_height = max(int(abs(top_right_after_rot[1] * 2) + 0.5), int(abs(bottom_right_after_rot[1] * 2) + 0.5))
        new_width = cols
        new_height = rows
        offset_x = (new_width - cols) / 2
        offset_y = (new_height - rows) / 2
        M[0, 2] += offset_x
        M[1, 2] += offset_y
        dst = cv2.warpAffine(img, M, (new_width, new_height))
        mask = cv2.warpAffine(mask, M, (new_width, new_height))
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

        return dst, mask
