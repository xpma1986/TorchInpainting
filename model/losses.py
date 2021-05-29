import torch
from torch.nn import Module
from torch.nn import L1Loss

from model.vgg16 import VGG16

class PerceptualLoss(Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
    
    def forward(self, vgg_out, vgg_gt, vgg_comp):
        loss = 0.
        for o, c, g in zip(vgg_out, vgg_comp, vgg_gt):
            loss += L1Loss()(o, g) + L1Loss()(c, g)

        return loss

class TotalVariationLoss(Module):
    def __init__(self, loss_weight=1):
        super(TotalVariationLoss, self).__init__()

        self.loss_weight = loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()

        return self.loss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

class ContentLoss(Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        
        self.loss_weights = [6.0, 1.0, 0.05, 1.0, 1.0, 0.1]
        self.vgg = VGG16()

        self.tv_loss = TotalVariationLoss()

    def forward(self, original, mask, inpainted):
        composite = mask*original + (1.0 - mask)*inpainted

        loss = []

        loss.append(L1Loss()((1.0-mask)*original, (1.0-mask)*inpainted))
        loss.append(L1Loss()(mask*original, mask*inpainted))

        vgg_out = self.vgg(inpainted)
        vgg_gt = self.vgg(original)
        vgg_comp = self.vgg(composite)

        loss.append(self.perceptual_loss(vgg_out, vgg_gt, vgg_comp))
        loss.append(self.style_loss(vgg_out, vgg_gt))
        loss.append(self.style_loss(vgg_comp, vgg_gt))

        loss.append(self.tv_loss(composite))

        total_loss = 0.

        for n in range(6):
            total_loss += self.loss_weights[n]*loss[n]

        return total_loss

    def perceptual_loss(self, vgg_out, vgg_gt, vgg_comp):
        loss = 0.
        for o, c, g in zip(vgg_out, vgg_comp, vgg_gt):
            loss += L1Loss()(o, g) + L1Loss()(c, g)

        return loss

    def style_loss(self, output, vgg_gt):
        return L1Loss()(self.gram_matrix(output), self.gram_matrix(vgg_gt))

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