""" Generate intermediate results using trained tocg """
""" intermediate results incoperate: fake warped cloth, fake warped clothmask, fake segmap """
import torch
import os
from HRVTDataset_test import HRVTDataset, CPDataLoader
from Configs.Config_intermediate import Config
from models import ConditionGenerator
from utils import load_checkpoint
from train_condition import one_iter as test_condition
from PIL import Image
from utils import visualize_segmap
from tqdm import tqdm


def to_img(tensor,mode):
    img = tensor[0].permute(1,2,0).detach().cpu().numpy()
    img = (img + 1)/2
    if mode == "P":
        img = img[:,:,0]
        
    return Image.fromarray((img*255).astype("uint8"),mode)


if __name__  == "__main__":
    
    opt = Config().get_opt()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    # dataset and dataloader
    dataset = HRVTDataset(opt,for_generator=False)
    dataloader = CPDataLoader(opt,dataset)
    
    # typemask
    if opt.type_mask:
        opt.semantic_nc += 1
        
    # tocg model
    tocg = ConditionGenerator(
        opt=opt,
        input1_nc=4,
        input2_nc=opt.semantic_nc + 3,
        output_nc=opt.output_nc,
        ngf=96,
        norm_layer=torch.nn.BatchNorm2d
    ).cuda()
    # load trained model
    load_checkpoint(tocg,opt.tocg_checkpoint,opt)

    # generate images with high resolution 
    opt.high_res = True
    
    for _ in tqdm(range(dataset.__len__())):
        
        # fetch data
        inputs = dataloader.next_batch()
        img_name = inputs["img_name"][0]

        # pass to model
        with torch.no_grad():
            tocg_outputs = test_condition(inputs,tocg, opt)
        fake_segmap, warped_cloth, warped_clothmask = tocg_outputs
        
        # png name
        img_name_seg = img_name.replace("jpg","png")
        # tensor name
        img_name_tensor = img_name.replace("jpg","pt")
        
        save_dir = f"{opt.output_root}/fake_segmap"
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        # fake segmentation map
        if opt.image_output:
            fake_segmap = visualize_segmap(fake_segmap, tensor_out=False)
            fake_segmap.save(f"{save_dir}/{img_name_seg}")
        else:
            torch.save(fake_segmap.cpu(), f"{save_dir}/{img_name_tensor}")
        
        save_dir = f"{opt.output_root}/fake_warped_cloth"
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        # warped_cloth
        if opt.image_output:
            warped_cloth = to_img(warped_cloth, "RGB")
            warped_cloth.save(f"{save_dir}/{img_name}")
        else:
            torch.save(warped_cloth.cpu(),  f"{save_dir}/{img_name_tensor}")
            
        save_dir = f"{opt.output_root}/fake_warped_clothmask"
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        # warped_cloth_mask
        if opt.image_output:
            warped_clothmask = to_img(warped_clothmask, "P")
            warped_clothmask.save(f"{save_dir}/{img_name_seg}")
        else:
            torch.save(warped_clothmask.cpu(),  f"{save_dir}/{img_name_tensor}")
            