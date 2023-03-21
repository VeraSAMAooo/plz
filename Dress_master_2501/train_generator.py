import os
import torch
from datetime import datetime
import torch.nn.functional as F
from HRVTDataset import HRVTDataset, CPDataLoader
from Configs.Config_condition_v3 import Config
from Configs.Config_generator_v4 import ConfigGenerator
from models import ConditionGenerator, SPADEGenerator, MultiscaleDiscriminator, define_D
from utils import VGGLoss
from utils import GANLossGenerator as GANLoss
from utils import create_network
from utils import load_checkpoint, visualize_generator
from sync_batchnorm import DataParallelWithCallback
import torchgeometry as tgm

def one_iter(inputs, tocg, generator, opt, gauss, unpaired=False):
    """ input data preparation """
    if not opt.GT:
        # train generator using fake segmentation maps
        from train_condition import one_iter as one_iter_condition
        opt.datamode = "pass"
        tocg.eval()
        with torch.no_grad():
            fake_segmap, warped_cloth, warped_clothmask = one_iter_condition(inputs, tocg, Config().get_opt())
            # make generator input parse map
            fake_parse_gauss = gauss(F.interpolate(fake_segmap, size=(opt.fine_height, opt.fine_width), mode='bilinear'))
            fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]

    else:   # train generators using ground true segmentation maps
        parse_GT = inputs["parse"].cuda()
        warped_cloth = inputs["warped_cloth"].cuda()
        # fake_parse = parse_GT.argmax(dim=1)[:, None].cuda()
        fake_parse_gauss = gauss(F.interpolate(parse_GT, size=(opt.fine_height, opt.fine_width), mode='bilinear'))
        fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]

    # map old segmentation map to new 7-channel segmentation map
    old_parse = torch.FloatTensor(fake_parse.size(0), opt.semantic_nc, opt.fine_height, opt.fine_width).zero_().cuda()
    old_parse.scatter_(1, fake_parse, 1.0)
    # for mozat dataset and bottom cloth
    labels = {
        0: ['background', [0]],
        1: ['hair', [1]],
        2: ['paste', [2, 8, 9, 10]],
        3: ['dress', [3]],
        4: ['left_arm', [4]],
        5: ['right_arm', [5]],
        6: ['left_leg', [6]],
        7: ['right_leg', [7]],
        8: ['noise', [11]]
    }

    parse = torch.FloatTensor(fake_parse.size(0), 9, opt.fine_height, opt.fine_width).zero_().cuda()
    for i in range(len(labels)):
        for label in labels[i][1]:
            parse[:, i] += old_parse[:, label]

    # if opt.type_mask:
    #     type_masks = torch.zeros_like(inputs["cloth_mask"]).cuda()
    #     for ii in range(opt.batch_size):
    #         if inputs["cloth_type"][ii] == "skirts":
    #             type_masks[ii] = 1
    #
    #     parse = torch.cat((parse,type_masks),dim=1)

    parse = parse.detach()
    agnostic = inputs["agnostic"].cuda()
    pose = inputs["densepose"].cuda()
    im = inputs["image"].cuda()

    generator_output = generator(torch.cat((agnostic, pose, warped_cloth), dim=1), parse)

    if opt.visualize:
        if opt.datamode == "train":
            visualize_generator(inputs["img_name"], torch.cat((inputs["image"].cuda(),generator_output),dim=0), opt.visualize_dir, opt.iter_idx)
            opt.visualize = False
        else:
            if not unpaired:
                opt.visual_names_paired += inputs["img_name"]
                opt.visual_tensors_paired.append(torch.cat((inputs["image"].cuda(),generator_output)))
                opt.visualize = False
            # else:
            #     opt.visual_names_unpaired += inputs["img_name"]
            #     opt.visual_tensors_unpaired.append(generator_output)
            #     opt.visualize = False

    if opt.datamode != "pass":

        """ train generator """
        fake_concat = torch.cat((parse, generator_output), dim=1)
        real_concat = torch.cat((parse, im), dim=1)
        pred = discriminator(torch.cat((fake_concat, real_concat), dim=0))
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            pred_fake = []
            pred_real = []
            for p in pred:
                pred_fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                pred_real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            pred_fake = pred[:pred.size(0) // 2]
            pred_real = pred[pred.size(0) // 2:]

        # losses for generator
        G_losses = {}
        # try it's best to cheat discriminator
        G_losses['GAN'] = criterionGAN(pred_fake, True, for_discriminator=False)

        if not opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = torch.cuda.FloatTensor(len(opt.gpu_ids)).zero_()
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not opt.no_vgg_loss:
            G_losses['VGG'] = criterionVGG(generator_output, im) * opt.lambda_vgg

        loss_gen = sum(G_losses.values()).mean()
        G_losses["G_total"] = loss_gen

        if opt.datamode == "train":
            optimizer_gen.zero_grad()
            loss_gen.backward()
            optimizer_gen.step()

        """ train discriminator """
        with torch.no_grad():
            output = generator(torch.cat((agnostic, pose, warped_cloth), dim=1), parse)
            output = output.detach()
            output.requires_grad_()

        fake_concat = torch.cat((parse, output), dim=1)
        real_concat = torch.cat((parse, im), dim=1)
        pred = discriminator(torch.cat((fake_concat, real_concat), dim=0))

        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            pred_fake = []
            pred_real = []
            for p in pred:
                pred_fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                pred_real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            pred_fake = pred[:pred.size(0) // 2]
            pred_real = pred[pred.size(0) // 2:]

        D_losses = {}
        D_losses['D_Fake'] = criterionGAN(pred_fake, False, for_discriminator=True)
        D_losses['D_Real'] = criterionGAN(pred_real, True, for_discriminator=True)

        loss_dis = sum(D_losses.values()).mean()
        D_losses["D_total"] = loss_dis

        if opt.datamode == "train":
            optimizer_dis.zero_grad()
            loss_dis.backward()
            optimizer_dis.step()

        for key in D_losses:
            G_losses[key] = D_losses[key]

        return G_losses
    else:
        return generator_output


def train():
    print("start training")
    # evaluation metric

    loss_train, loss_val = None, None
    # print interval
    interval = opt.ckpt_num // opt.batch_size

    for iter_idx in range(opt.load_step, opt.keep_step):

        opt.iter_idx = iter_idx

        # for visualization
        if (iter_idx-opt.load_step) % interval == 0:
            opt.visualize = True
            opt.visualize_dir = "./visualization/generator_ori/train"

        opt.datamode = "train"
        generator.train()
        discriminator.train()
        train_inputs = train_loader.next_batch()
        cur_loss = one_iter(train_inputs,tocg, generator, opt, gauss)
        loss_train = loss_update(loss_train, cur_loss)

        if (iter_idx-opt.load_step) % interval == 0:

            loss_val = validate()

            # save model
            model_name = str(iter_idx) + ".pth"
            torch.save(generator.state_dict(),os.path.join(G_save_path,model_name))
            torch.save(discriminator.state_dict(),os.path.join(D_save_path,model_name))

            # log & print
            message_train = '(step: %d/%d)[train] ' % (iter_idx,opt.keep_step)
            message_val = '(step: %d/%d)[val] ' % (iter_idx,opt.keep_step)
            for k, v in loss_train.items():
                message_train += '%s: %.3f ' % (k, v/interval)
            for k, v in loss_val.items():
                message_val += '%s: %.3f ' % (k, v)

            print(message_train+"\n"+message_val)

            with open(log_path, "a") as log_file:
                log_file.write('%s\n' % message_train)
                log_file.write('%s\n' % message_val)
            
            loss_train = None

def validate():
    # validation
    opt.datamode = "test"
    generator.eval()
    discriminator.eval()

    # visualization
    visual_interval = opt.val_num // 4
    opt.visual_names_paired = []
    # opt.visual_names_unpaired = []
    opt.visual_tensors_paired = []
    # opt.visual_tensors_unpaired = []

    loss_val = None
    with torch.no_grad():
        for iter_idx in range(opt.val_num // opt.batch_size):
            if iter_idx % visual_interval == 0:
                opt.visualize = True
                opt.visualize_dir_paired = "./visualization/generator/val"
                # opt.visualize_dir_unpaired = "./visualization/generator/val_unpaired"
            val_inputs_paired = val_loader.next_batch()
            cur_loss = one_iter(val_inputs_paired, tocg, generator, opt, gauss)
            # _ = one_iter(val_inputs_unpaired, tocg, generator, opt, gauss, unpaired=True)
            loss_val = loss_update(loss_val, cur_loss)

    visualize_generator(opt.visual_names_paired, opt.visual_tensors_paired, opt.visualize_dir_paired, opt.iter_idx)
    # visualize_generator(opt.visual_names_unpaired, opt.visual_tensors_unpaired, opt.visualize_dir_unpaired, opt.iter_idx)

    # del opt.visual_names
    # del opt.visual_tensors
    # opt.visual_names = []
    # opt.visual_tensors = []

    for key in loss_val:
        loss_val[key] /= iter_idx
    return loss_val

def loss_update(loss_total, cur_loss):
    if not loss_total:
        for key in cur_loss:
            cur_loss[key] = cur_loss[key].mean()
        return cur_loss

    else:
        for key in loss_total.keys():
            loss_total[key] += cur_loss[key].mean()
        return loss_total

if __name__ == "__main__":

    """ configurations """
    opt = ConfigGenerator().get_opt()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids

    """ dataset & dataloader """
    opt.datamode = "train"
    train_set = HRVTDataset(opt,for_generator=True)
    train_loader = CPDataLoader(opt, train_set)
    opt.datamode = "test"
    val_set = HRVTDataset(opt,for_generator=True)
    val_loader = CPDataLoader(opt, val_set)

    """ model """
    tocg = None

    # load weights for tocg
    if not opt.tocg_checkpoint == '' and os.path.exists(opt.tocg_checkpoint):
        load_checkpoint(tocg, opt.tocg_checkpoint, opt)

    # try on image generator
    if opt.type_mask:
        opt.gen_semantic_nc += 1
        opt.dis_input_nc += 1
    generator = SPADEGenerator(opt, 3 + 3 + 3).cuda()
    # initialize weights for generator
    generator.init_weights(opt.init_type, opt.init_variance)

    # discriminator
    # discriminator = create_network(MultiscaleDiscriminator, opt)
    discriminator = define_D(opt.dis_input_nc, gpu_ids=opt.gpu_ids, getIntermFeat=not opt.no_ganFeat_loss)

    """ criterions """
    criterionGAN = GANLoss('hinge', tensor=torch.cuda.FloatTensor)
    criterionFeat = torch.nn.L1Loss()
    criterionVGG = VGGLoss(opt)


    # """ Parallel Training """
    # if len(opt.gpu_ids) > 0:
    #     # tocg = DataParallelWithCallback(tocg, device_ids=opt.gpu_ids)
    #     generator = DataParallelWithCallback(generator, device_ids=opt.gpu_ids)
    #     discriminator = DataParallelWithCallback(discriminator, device_ids=opt.gpu_ids)
    #     criterionGAN = DataParallelWithCallback(criterionGAN, device_ids=opt.gpu_ids)
    #     criterionFeat = DataParallelWithCallback(criterionFeat, device_ids=opt.gpu_ids)
    #     criterionVGG = DataParallelWithCallback(criterionVGG, device_ids=opt.gpu_ids)

    # load weights for generator
    if not opt.generator_checkpoint == '' and os.path.exists(opt.generator_checkpoint):
        load_checkpoint(generator, opt.generator_checkpoint, opt)

    if not opt.generator_D_checkpoint == '' and os.path.exists(opt.generator_D_checkpoint):
        load_checkpoint(discriminator, opt.generator_D_checkpoint, opt)

    """ optimizers """
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=opt.G_lr, betas=(0, 0.9))
    scheduler_gen = torch.optim.lr_scheduler.LambdaLR(optimizer_gen, lr_lambda=lambda step: 1.0 - max(0, step * 1000 + opt.load_step - opt.keep_step) / float(opt.decay_step + 1))

    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=opt.D_lr, betas=(0, 0.9))
    scheduler_dis = torch.optim.lr_scheduler.LambdaLR(optimizer_dis, lr_lambda=lambda step: 1.0 - max(0, step * 1000 + opt.load_step - opt.keep_step) / float(opt.decay_step + 1))

    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    gauss = gauss.cuda()

    # logging
    log_path = os.path.join(opt.log_root,opt.log_generator)
    if not os.path.exists(log_path): os.makedirs(log_path)
    time = str(datetime.now().strftime("%m_%d_%Y_%H_%M"))
    log_path = os.path.join(log_path,time) + ".txt"
    with open(log_path,"w") as ff:
        ff.write("start logging ... \n")
        ff.close()

    # model checkpoint
    # if not os.path.exists(opt.checkpoint_root): os.mkdir(opt.checkpoint_root)
    G_save_path = os.path.join(opt.checkpoint_root,opt.generator_savedir)
    if not os.path.exists(G_save_path): os.makedirs(G_save_path)

    D_save_path = os.path.join(opt.checkpoint_root,opt.generator_D_savedir)
    if not os.path.exists(D_save_path): os.makedirs(D_save_path)

    train()














