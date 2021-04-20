import torch
from torch.nn import Module
from torch.nn import L1Loss

from model.vgg16 import VGG16

class PerceptualLoss(Module):
    def __init__(self):
        return
    
    def forward(self, vgg_out, vgg_gt, vgg_comp):
        loss = 0.
        for o, c, g in zip(vgg_out, vgg_comp, vgg_gt):
            loss += L1Loss()(o, g) + L1Loss()(c, g)

        return loss

class ContentLoss(Module):
    def __init__(self):
        self.weights = [6.0, 1.0, 0.05, 1.0, 1.0, 0.1]
        self.vgg = VGG16()

    def forward(self, original, mask, inpainted):
        composite = mask*original + (1.0 - mask)*inpainted
        l1 = L1Loss()((1.0-mask)*original, (1.0-mask)*inpainted)
        l2 = L1Loss()(mask*original, mask*inpainted)

        vgg_out = self.vgg(inpainted)
        vgg_gt = self.vgg(original)
        vgg_comp = self.vgg(composite)


        return self.weights[0]*self.L1()

    def perceptual_loss(self, vgg_out, vgg_gt, vgg_comp):
        loss = 0.
        for o, c, g in zip(vgg_out, vgg_comp, vgg_gt):
            loss += L1Loss()(o, g) + L1Loss()(c, g)

        return loss

    def style_loss(self, output, vgg_gt):
        return

    def gram_matrix(tensor):
        #Unwrapping the tensor dimensions into respective variables i.e. batch size, distance, height and width 
        _, d, h, w=tensor.size() 
        #Reshaping data into a two dimensional of array or two dimensional of tensor
        tensor=tensor.view(d, h*w)
        #Multiplying the original tensor with its own transpose using torch.mm 
        #tensor.t() will return the transpose of original tensor
        gram=torch.mm(tensor, tensor.t())
        #Returning gram matrix 
        return gram