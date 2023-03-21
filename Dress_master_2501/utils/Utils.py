import os
import torch
import functools
import torch.nn as nn
from PIL import Image


def make_grid(N, iH, iW,opt):
    grid_x = torch.linspace(-1.0, 1.0, iW).view(1, 1, iW, 1).expand(N, iH, -1, -1)
    grid_y = torch.linspace(-1.0, 1.0, iH).view(1, iH, 1, 1).expand(N, -1, iW, -1)
    if opt.cuda:
        grid = torch.cat([grid_x, grid_y], 3).cuda()
    else:
        grid = torch.cat([grid_x, grid_y], 3)
    return grid


def load_checkpoint(model, checkpoint_path, opt):
    if not os.path.exists(checkpoint_path):
        print('no checkpoint')
        raise
    log = model.load_state_dict(torch.load(checkpoint_path), strict=True)
    if opt.cuda :
        model.cuda()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def iou_metric(y_pred_batch, y_true_batch):
    B = y_pred_batch.shape[0]
    iou = 0
    for i in range(B):
        y_pred = y_pred_batch[i]
        y_true = y_true_batch[i]
        # y_pred is not one-hot, so need to threshold it
        y_pred = y_pred > 0.5

        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        intersection = torch.sum(y_pred[y_true == 1])
        union = torch.sum(y_pred) + torch.sum(y_true)

        iou += (intersection + 1e-7) / (union - intersection + 1e-7) / B
    return iou


def remove_overlap(seg_out, warped_cm):
    assert len(warped_cm.shape) == 4

    warped_cm = warped_cm - (torch.cat([seg_out[:, 1:3, :, :], seg_out[:, 5:, :, :]], dim=1)).sum(dim=1,
                                                                                                  keepdim=True) * warped_cm
    return warped_cm

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def visualize_generator(img_names, gen_outputs, save_dir, iter_idx):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    img_names = [name.replace(".jpg","") for name in img_names]

    if type(gen_outputs) == list:
        gen_outputs = torch.cat(gen_outputs,dim=0)

    imgs = []

    for idx in range(gen_outputs.shape[0]):
        img = gen_outputs[idx].permute(1,2,0).detach().cpu().numpy()
        img = img/2+0.5
        img = Image.fromarray((img*255).astype("uint8"))
        imgs.append(img)

    num_row = int(len(imgs)**0.5)
    num_col = len(imgs) // num_row
    if num_row * num_col != len(imgs):
        num_col += 1

    grid = image_grid(imgs, num_row, num_col)
    img_name = str(iter_idx) + "_"  + "_".join(img_names)
    save_path = os.path.join(save_dir,img_name) + ".jpg"
    grid.save(save_path)
    del grid


def visualize_condition(img_names, gen_outputs, save_dir, iter_idx, num_row):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    img_names = [name.replace(".jpg","") for name in img_names]

    if type(gen_outputs) == list:
        gen_outputs = torch.cat(gen_outputs,dim=0)

    imgs = []

    for idx in range(gen_outputs.shape[0]):
        img = gen_outputs[idx].permute(1,2,0).detach().cpu().numpy()
        img = img/2+0.5
        img = Image.fromarray((img*255).astype("uint8"))
        imgs.append(img)

    # num_row = int(len(imgs)**0.5)
    num_col = len(imgs) // num_row
    if num_row * num_col != len(imgs):
        num_col += 1

    grid = image_grid(imgs, num_row, num_col)
    img_name = str(iter_idx) + "_"  + "_".join(img_names)
    save_path = os.path.join(save_dir,img_name) + ".jpg"
    grid.save(save_path)
    del grid


    
