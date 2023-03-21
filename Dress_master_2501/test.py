import torch
import os
import argparse
import torch.nn.functional as F
from HRVTDataset_test import HRVTDataset, CPDataLoader
from Configs.Config_test_v1 import ConfigTest
from models import ConditionGenerator, SPADEGenerator
from utils import load_checkpoint
from train_condition import one_iter as test_condition
from train_generator import one_iter as test_generator
from PIL import Image
import PIL
from utils import image_grid, visualize_segmap
from tqdm import tqdm
import torchgeometry as tgm
from sync_batchnorm import DataParallelWithCallback


def get_cmd_opt():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass_list", dest="pass_list")
    parser.add_argument("--output_root", dest="output_root")
    parser.add_argument("--tocg_model", dest="tocg_checkpoint")
    args = parser.parse_args()
    return args


def to_img(tensor, mode="RGB"):
    
    if type(tensor) == PIL.Image.Image:
        return tensor
    
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    img = tensor.permute(1,2,0).detach().cpu().numpy()
    img = (img + 1)/2
    if mode == "P":
        img = img[:,:,0]
    return Image.fromarray((img*255).astype("uint8"),mode)


def visual_tocg_outputs(tocg_outputs):
    # parse condition outputs
    # use segmentation map predicted by condition generator
    fake_segmap, warped_cloth, warped_clothmask = tocg_outputs

    visual_segmap = visualize_segmap(fake_segmap, tensor_out=False)
    
    # warped_cloth
    visual_warped_cloth = to_img(warped_cloth, "RGB")
    
    # warped_cloth_mask
    visual_warped_clothmask = to_img(warped_clothmask, "P")

    return visual_segmap, visual_warped_cloth, visual_warped_clothmask

def tocg_downsample(ori_inputs):

    new_inputs = {}
    for key, val in ori_inputs.items():
        if key in ["cloth","densepose"]:
            new_inputs[key] = F.interpolate(val, size=(256, 192), mode='bilinear')
        elif key in ["cloth_mask","parse_agnostic"]:
            new_inputs[key] = F.interpolate(val, size=(256, 192), mode='nearest')
        else:
            new_inputs[key] = val

    return new_inputs


if __name__ == "__main__":
    
    """ Configuration """
    opt = ConfigTest().get_opt()
    args = get_cmd_opt()
    opt.output_root = args.output_root
    opt.pass_list = args.pass_list
    opt.tocg_checkpoint = args.tocg_checkpoint
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    
    """ Data Preparation """
    print("Preparing Data...")
    opt.fine_height, opt.fine_width = opt.generator_fine_height, opt.generator_fine_width
    dataset = HRVTDataset(opt,for_generator=True)
    # turn off shuffle mode for dataloader
    opt.shuffle = False
    dataloader = CPDataLoader(opt, dataset)


    """ Models """
    print("Loading models...")
    # Try-On Condition Generator
    # opt.fine_height, opt.fine_width = opt.tocg_fine_height, opt.tocg_fine_width
    if opt.type_mask_condition:
        opt.semantic_nc += 1
    tocg = ConditionGenerator(
        opt=opt,
        input1_nc=4,
        input2_nc=opt.semantic_nc + 3,
        output_nc=opt.output_nc,
        ngf=96,
        norm_layer=torch.nn.BatchNorm2d
    ).cuda()
    load_checkpoint(tocg,opt.tocg_checkpoint,opt)
    
    # Try-On Imge Generator
    if opt.type_mask_generator:
        opt.gen_semantic_nc += 1
        opt.dis_input_nc += 1
    generator = SPADEGenerator(opt, 3 + 3 + 3).cuda()

    if opt.data_paralle:
        generator = DataParallelWithCallback(generator, device_ids=opt.gpu_ids)
    load_checkpoint(generator,opt.generator_checkpoint,opt)

    # Gaussian Noise
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    gauss = gauss.cuda()

    """ Main Loop """
    print("Predicting...")
    for _ in tqdm(range(dataset.__len__())):
        # fetch data for one batch
        inputs = dataloader.next_batch()

        """ Pass to Try-on Condition Generator using origional inputs """
        opt.high_res = True # for high-resolution composition
        opt.fine_height, opt.fine_width = opt.tocg_fine_height, opt.tocg_fine_width
        # tocg_inputs = crop_and_resize(opt,inputs,crop_factor=0.)
        opt.type_mask = opt.type_mask_condition
        with torch.no_grad():
            tocg_outputs = test_condition(tocg_downsample(inputs), tocg, opt)
        opt.fine_height, opt.fine_width = opt.generator_fine_height, opt.generator_fine_width
  
        fake_segmap, warped_cloth, warped_clothmask = tocg_outputs
        # warped_cloth.unsqueeze_(0)
        # warped_clothmask.unsqueeze_(0)
        # fake_segmap.unsqueeze_(0)

        """ Save Results for tocg """
        for idx in range(len(inputs["img_name"])):
            img_name, cloth_name = inputs["img_name"][idx], inputs["cloth_name"][idx]
            model_truth, cloth_truth = inputs["image"][idx], inputs["cloth"][idx]
            visual_segmap, visual_warped_cloth, _ = visual_tocg_outputs(tocg_outputs)
            imgs = list(map(to_img,[model_truth, cloth_truth, visual_segmap, visual_warped_cloth]))
            grid = image_grid(imgs, 1, 4)
            file_name = "_".join([img_name, cloth_name])
            file_root = os.path.join(opt.output_root, "tocg")
            if not os.path.exists(file_root): os.makedirs(file_root)
            grid.save(os.path.join(file_root,file_name))

        if not opt.GT_generator_input:
            inputs["parse"] = fake_segmap
            # use warped cloth predicted by condition generator
            inputs["warped_cloth"] = warped_cloth
            # use warped clothmask predicted by condition generator
            inputs["warped_cloth_mask"] = warped_clothmask
        else:
            image_name = inputs["img_name"][0].replace("jpg","pt")
            inputs["parse"] = torch.load(os.path.join(opt.imgroot,"fake_segmap",image_name)).cuda()
            inputs["warped_cloth"] = torch.load(os.path.join(opt.imgroot,"fake_warped_cloth",image_name)).cuda()
            inputs["warped_cloth_mask"] = torch.load(os.path.join(opt.imgroot,"fake_warped_clothmask",image_name)).cuda()

        # generator_inputs = crop_and_resize(opt,inputs,crop_factor=0.)

        """ Pass to Try-on Image Generator using refined inputs """
        opt.type_mask = opt.type_mask_generator
        with torch.no_grad():
            generator_output = test_generator(inputs, tocg, generator, opt, gauss)

        """ Save Results for Generator """
        for idx in range(len(inputs["img_name"])):
            img_name, cloth_name = inputs["img_name"][idx], inputs["cloth_name"][idx]
            model_truth, cloth_truth, warped_cloth, fake_img = inputs["image"][idx], inputs["cloth"][idx], inputs['warped_cloth'][idx], generator_output[idx]
            imgs = list(map(to_img,[model_truth, cloth_truth, warped_cloth, fake_img]))
            grid = image_grid(imgs, 1, 4)
            file_name = "_".join([img_name, cloth_name])
            file_root = os.path.join(opt.output_root,"generator")
            if not os.path.exists(file_root): os.makedirs(file_root)
            grid.save(os.path.join(file_root,file_name))