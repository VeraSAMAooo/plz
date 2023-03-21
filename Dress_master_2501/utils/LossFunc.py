import torch
import torch.nn as nn
from .VGG19 import Vgg19
from torch.autograd import Variable
import torch.nn.functional as F


class VGGLoss(nn.Module):
    def __init__(self, opt, layids=None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        if opt.cuda:
            self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class GANLossGenerator(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super().__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


def compute_tv_loss(opt, flow_list, warped_clothmask):

    loss_tv = 0

    if opt.edgeawaretv == 'no_edge':
        if not opt.lasttvonly:
            for flow in flow_list:
                y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                loss_tv = loss_tv + y_tv + x_tv
        else:
            for flow in flow_list[-1:]:
                y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                loss_tv = loss_tv + y_tv + x_tv
    else:
        if opt.edgeawaretv == 'last_only':
            flow = flow_list[-1]
            warped_clothmask_paired_down = F.interpolate(warped_clothmask, flow.shape[1:3], mode='bilinear')
            y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :])
            x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
            mask_y = torch.exp(-150 * torch.abs(
                warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, 1:, :, :] - warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :-1, :, :]))
            mask_x = torch.exp(-150 * torch.abs(
                warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :, 1:, :] - warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :, :-1, :]))
            y_tv = y_tv * mask_y
            x_tv = x_tv * mask_x
            y_tv = y_tv.mean()
            x_tv = x_tv.mean()
            loss_tv = loss_tv + y_tv + x_tv

        elif opt.edgeawaretv == 'weighted':
            for i in range(5):
                flow = flow_list[i]
                warped_clothmask_paired_down = F.interpolate(warped_clothmask, flow.shape[1:3], mode='bilinear')
                y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :])
                x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
                mask_y = torch.exp(-150 * torch.abs(warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, 1:, :, :] - warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :-1, :, :]))
                mask_x = torch.exp(-150 * torch.abs(warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :, 1:, :] - warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :, :-1, :]))
                y_tv = y_tv * mask_y
                x_tv = x_tv * mask_x
                y_tv = y_tv.mean() / (2 ** (4 - i))
                x_tv = x_tv.mean() / (2 ** (4 - i))
                loss_tv = loss_tv + y_tv + x_tv

        if opt.add_lasttv:
            for flow in flow_list[-1:]:
                y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                loss_tv = loss_tv + y_tv + x_tv

    return loss_tv


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss
