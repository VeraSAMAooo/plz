import os
import torch
import collections
import numpy as np
from datetime import datetime
import torch.nn.functional as F
from HRVTDataset import HRVTDataset, CPDataLoader
from Configs.Config_condition_v5 import Config
from models import ConditionGenerator, define_D
from utils import GANLoss, VGGLoss, composition, compute_tv_loss, cross_entropy2d
from utils import make_grid as mkgrid
from utils import remove_overlap, load_checkpoint, iou_metric, visualize_condition, visualize_segmap


def one_iter(inputs, tocg, opt):
    # input1
    cloth = inputs["cloth"].cuda()
    cloth_mask = inputs["cloth_mask"].cuda()
    cloth_mask = torch.FloatTensor((cloth_mask.detach().cpu().numpy() > 0.5).astype(np.float32)).cuda()
    # input2
    parse_agnostic = inputs["parse_agnostic"].cuda()
    densepose = inputs["densepose"].cuda()
    # GT
    if opt.datamode != "pass":
        label_onehot = inputs['parse_onehot'].cuda()  # CE
        label = inputs['parse'].cuda()  # GAN loss
        parse_cloth_mask = inputs['warped_cloth_mask'].cuda()  # L1
    im_c = inputs['warped_cloth'].cuda()  # VGG

    # type mask
    # if opt.type_mask:

    # tocg inputs
    input1 = torch.cat([cloth, cloth_mask], 1)
    input2 = torch.cat([parse_agnostic, densepose], 1)


    # forward
    tocg_outputs = tocg(opt, input1, input2)
    flow_list, fake_segmap, warped_cloth, warped_clothmask = tocg_outputs

    # composition
    warped_cloth, warped_clothmask, fake_segmap = composition(opt, tocg_outputs, inputs)

    if opt.visualize:

        visual_segmaps = [visualize_segmap(fake_segmap,batch=i)[None,:].cuda() for i in range(opt.batch_size)] 
        visual_segmaps = torch.cat(visual_segmaps, dim=0)

        if opt.datamode == "train":
            visual_tensor = torch.cat((inputs["image"].cuda(), cloth, warped_cloth, visual_segmaps, im_c),dim=0)
            visualize_condition(inputs["img_name"], visual_tensor, "./visualization/tocg/train", opt.iter_idx, 5)

        if opt.datamode == "test" and opt.paired:
            visual_tensor = torch.cat((inputs["image"].cuda(), cloth, warped_cloth, visual_segmaps, im_c),dim=0)
            visualize_condition(inputs["img_name"], visual_tensor, "./visualization/tocg/test", opt.iter_idx, 5)

        if opt.datamode == "test" and not opt.paired:
            visual_tensor = torch.cat((inputs["image"].cuda(), cloth, warped_cloth, visual_segmaps),dim=0)
            visualize_condition(inputs["img_name"], visual_tensor, "./visualization/tocg/test_unpaired", opt.iter_idx, 4)

        opt.visualize = False

    # compute misalignment
    if opt.datamode != "pass":
        # loss warping
        # l1 loss without first term
        loss_l1_cloth = criterionL1(warped_clothmask, parse_cloth_mask)
        # vgg loss without first term
        loss_vgg = criterionVGG(warped_cloth, im_c)
        # add second term to l1 loss and vgg loss
        N, _, iH, iW = cloth.size()
        # Intermediate flow loss
        if opt.interflowloss:
            for i in range(len(flow_list) - 1):
                flow = flow_list[i]
                N, fH, fW, _ = flow.size()
                grid = mkgrid(N, iH, iW, opt)
                flow = F.interpolate(flow.permute(0, 3, 1, 2), size=cloth.shape[2:], mode=opt.upsample).permute(0, 2, 3, 1)
                flow_norm = torch.cat([flow[:, :, :, 0:1] / ((fW - 1.0) / 2.0), flow[:, :, :, 1:2] / ((fH - 1.0) / 2.0)], 3)
                warped_c = F.grid_sample(cloth, flow_norm + grid, padding_mode='border')
                warped_cm = F.grid_sample(cloth_mask, flow_norm + grid, padding_mode='border')
                warped_cm = remove_overlap(F.softmax(fake_segmap, dim=1), warped_cm)
                loss_l1_cloth += criterionL1(warped_cm, parse_cloth_mask) / (2 ** (4 - i))
                loss_vgg += criterionVGG(warped_c, im_c) / (2 ** (4 - i))
        # tv loss
        loss_tv = compute_tv_loss(opt, flow_list, warped_clothmask)
        # loss segmentation
        CE_loss = cross_entropy2d(fake_segmap, label_onehot.transpose(0, 1)[0].long())
        # total loss
        loss_G = (10 * loss_l1_cloth + loss_vgg + opt.tvlambda * loss_tv) + (CE_loss * opt.CElamda)

        errors = {
            "l1_cloth": loss_l1_cloth.item(), 
            "vgg": loss_vgg.item(), 
            "tv": loss_tv.item(), 
            "CE": CE_loss.item(), 
            "loss_G": loss_G.item()
        }

        if opt.datamode == "train":

            if opt.no_GAN_loss: 

                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()
                return errors

            else:   # train condition generator as a GAN
                
                fake_segmap_softmax = torch.softmax(fake_segmap, 1)

                pred_segmap = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax), dim=1))
                
                loss_G_GAN = criterionGAN(pred_segmap, True)

                # train generator and discriminator separately

                loss_G += loss_G_GAN * opt.GANlambda

                # step generator
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

                # discriminator
                with torch.no_grad():
                    _, fake_segmap, _, _ = tocg(opt, input1, input2)
                fake_segmap_softmax = torch.softmax(fake_segmap, 1)
                
                # loss discriminator
                fake_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax.detach()),dim=1))
                real_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), label),dim=1))
                loss_D_fake = criterionGAN(fake_segmap_pred, False)
                loss_D_real = criterionGAN(real_segmap_pred, True)
                
                loss_D = loss_D_fake + loss_D_real
                
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()   

                errors_with_gan = {
                    "l1_cloth": loss_l1_cloth.item(), 
                    "vgg": loss_vgg.item(), 
                    "tv": loss_tv.item(), 
                    "CE": CE_loss.item(), 
                    "loss_D_real": loss_D_real.item(),
                    "loss_D_fake": loss_D_fake.item(),
                    "loss_D": loss_D.item(),
                    "loss_G": loss_G.item()
                }           

                return errors_with_gan

        elif opt.datamode == "test" and opt.paired:
            iou = iou_metric(F.softmax(fake_segmap, dim=1).detach(), label)
            errors["iou"] = iou
            return errors

    else:
        return fake_segmap, warped_cloth, warped_clothmask


def validate():
    # validation
    opt.datamode = "test"
    tocg.eval()
    # D.eval()

    val_metrics = collections.defaultdict(float)

    if opt.paired:
        num_batch = opt.val_num // opt.batch_size 
    else: # only for visualization, so we only check one batch 
        num_batch = 1

    with torch.no_grad():
        for iter_idx in range(num_batch):

            if iter_idx == 0:
                opt.visualize = True

            if opt.paired:
                val_inputs, _ = val_loader.next_batch()
            else:
                _, val_inputs = val_loader.next_batch()

            cur_metrics = one_iter(val_inputs,tocg,opt)
            if not opt.paired:
                return 
            # update validation metric
            for key, value in cur_metrics.items():
                val_metrics[key] += value

    for key in val_metrics:
        val_metrics[key]/=iter_idx
    return val_metrics


def train():
    print("start training")
    # evaluation metric
    opt.visualize = False
    # initialize error dictionary
    errors = collections.defaultdict(float)

    interval = opt.ckpt_num // opt.batch_size

    for iter_idx in range(opt.load_step, opt.keep_step):
        # print interval
        opt.datamode = "train"
        tocg.train()
        # D.train()

        if (iter_idx-opt.load_step) % interval == 0:
            opt.visualize = True
            opt.iter_idx = iter_idx

        train_inputs, _ = train_loader.next_batch()
        cur_errors = one_iter(train_inputs, tocg, opt)

        # for recording
        for key, value in cur_errors.items():
            errors[key] += value

        if (iter_idx-opt.load_step) % interval == 0:

            # validate 
            val_metrics = validate()
            # visualize unpaired results
            opt.paired = False
            validate()
            opt.paired = True
            
            # save model
            model_name = str(iter_idx) + ".pth"
            torch.save(tocg.state_dict(),os.path.join(tocg_save_path,model_name))
            if not opt.no_GAN_loss:
                torch.save(D.state_dict(),os.path.join(D_save_path,model_name))

            # log & print
            message_train = '(step: %d/%d)[train] ' % (iter_idx,opt.keep_step)
            message_val = '(step: %d/%d)[val] ' % (iter_idx,opt.keep_step)
            for k, v in errors.items():
                message_train += '%s: %.3f ' % (k, v/interval)
            for k, v in val_metrics.items():
                message_val += '%s: %.3f ' % (k, v)

            print(message_train+"\n"+message_val)

            with open(log_path, "a") as log_file:
                log_file.write('%s\n' % message_train)
                log_file.write('%s\n' % message_val)

            for key in errors.keys():
                errors[key] = 0.
    
    print("Finished")


if __name__ == "__main__":
    """ configurations """
    opt = Config().get_opt()
    # GPU configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids

    opt.batch_size = opt.condition_batch_size

    """ dataset & dataloader """
    opt.datamode = "train"
    train_set = HRVTDataset(opt)
    train_loader = CPDataLoader(opt, train_set)
    opt.datamode = "test"
    val_set = HRVTDataset(opt)
    val_loader = CPDataLoader(opt, val_set)

    """ model """
    # try on condition generator
    # additional channel for type mask
    if opt.type_mask:
        opt.semantic_nc += 1
    tocg = ConditionGenerator(
        opt=opt,
        input1_nc=4,
        input2_nc=opt.semantic_nc + 3,
        output_nc=opt.output_nc,
        ngf=96,
        norm_layer=torch.nn.BatchNorm2d
    ).cuda()

    # discriminator
    D = define_D(
        input_nc=4 + opt.semantic_nc + 3 + opt.output_nc,
        Ddownx2=opt.Ddownx2,
        Ddropout=opt.Ddropout,
        n_layers_D=3,
        spectral=opt.spectral,
        num_D=opt.num_D
        ).cuda()

    """ load checkpoint """
    if not opt.tocg_checkpoint == '' and os.path.exists(opt.tocg_checkpoint):
        load_checkpoint(tocg, opt.tocg_checkpoint,opt)

    """ criterion """
    criterionL1 = torch.nn.L1Loss()
    criterionVGG = VGGLoss(opt)
    criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor if opt.gpu_ids else torch.Tensor)

    """ optimizers """
    optimizer_G = torch.optim.Adam(tocg.parameters(), lr=opt.tocg_lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.D_lr, betas=(0.5, 0.999))

    # logging
    loss_record = dict()
    if not os.path.exists(opt.log_root): os.mkdir(opt.log_root)
    log_path = os.path.join(opt.log_root,opt.log_condition)
    if not os.path.exists(log_path): os.mkdir(log_path)
    time = str(datetime.now().strftime("%m_%d_%Y_%H_%M"))
    log_path = os.path.join(log_path,time) + ".txt"
    with open(log_path,"w") as ff:
        ff.write("start logging ... \n")
        ff.close()
    
    # model checkpoint
    tocg_save_path = os.path.join(opt.checkpoint_root,opt.tocg_savedir)
    if not os.path.exists(tocg_save_path): os.makedirs(tocg_save_path)
    if not opt.no_GAN_loss:
        D_save_path = os.path.join(opt.checkpoint_root,opt.tocg_D_savedir)
        if not os.path.exists(D_save_path): os.makedirs(D_save_path)

    # train
    train()





