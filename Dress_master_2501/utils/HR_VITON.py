import torch
import numpy as np
from .Utils import remove_overlap, make_grid
import torch.nn.functional as F


def composition(opt, tocg_outputs, inputs=None):
    flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg_outputs

    warped_cm_onehot = torch.FloatTensor((warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
    # fake segmap cloth channel * warped clothmask
    if opt.clothmask_composition != 'no_composition':
        if opt.clothmask_composition == 'detach':
            cloth_mask = torch.ones_like(fake_segmap.detach())
            cloth_mask[:, 4:5, :, :] = warped_cm_onehot
            fake_segmap = fake_segmap * cloth_mask

        if opt.clothmask_composition == 'warp_grad':
            cloth_mask = torch.ones_like(fake_segmap.detach())
            cloth_mask[:, 4:5, :, :] = warped_clothmask_paired
            fake_segmap = fake_segmap * cloth_mask

    if opt.high_res:
        # warped cloth
        N, iH, iW = opt.batch_size, 512, 384
        fake_segmap = F.interpolate(fake_segmap, size=(iH, iW), mode='bilinear')
        cloth = inputs["cloth"].cuda()
        cloth_mask = inputs["cloth_mask"].cuda()
        grid = make_grid(N, iH, iW,opt)
        flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
        flow_norm = torch.cat([flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)], 3)
        warped_grid = grid + flow_norm
        warped_cloth_paired = F.grid_sample(cloth, warped_grid, padding_mode='border').detach()
        warped_clothmask_paired = F.grid_sample(cloth_mask, warped_grid, padding_mode='border')


    if opt.occlusion:
        warped_clothmask_paired = remove_overlap(F.softmax(fake_segmap, dim=1), warped_clothmask_paired)
        warped_cloth_paired = warped_cloth_paired * warped_clothmask_paired + torch.ones_like(warped_cloth_paired) * (
                    1 - warped_clothmask_paired)

    return warped_cloth_paired, warped_clothmask_paired, fake_segmap