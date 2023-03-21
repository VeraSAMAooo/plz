import os
import torch
import json
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode as im
from PIL import Image
import numpy as np


def crop_and_resize(opt,ori_result,crop_factor=0.2):

    # result = {}
    # top, left = int(opt.input_height*crop_factor), int(opt.input_width*crop_factor/2)
    # height, width = opt.input_height-top, opt.input_width-2*left
    # crop_size = (top,left,height,width)
    #
    # for key, val in ori_result.items():
    #     if key == "img_name" or key == "cloth_name" or key == "cloth_type":
    #         result[key] = val
    #         continue
    #
    #     if len(val.shape) < 2:
    #         continue
    #     val = transforms.functional.crop(val,*crop_size)
    #     val = transforms.functional.resize(val, (opt.fine_height, opt.fine_width))
    #     result[key] = val

    # return result
    return ori_result


class HRVTDataset(Dataset):
    """ Dataset for HR-VITON"""
    def __init__(self, opt, for_generator=False):
        super().__init__()
        # base setting
        self.for_generator = for_generator
        self.opt = opt
        self.datamode = opt.datamode    # train or test or self-defined
        self.imgroot = opt.imgroot
        self.clothroot = opt.clothroot
        self.input_height = opt.input_height
        self.input_width = opt.input_width
        self.semantic_nc = opt.semantic_nc
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                # normalize each channel with mean = 0.5 and std = 0.5
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.datalist = opt.pass_list

        with open(os.path.join(self.datalist),"r") as ff:
            self.datalist = ff.readlines()
            ff.close()
        
        # with open(opt.type_dict_path,'r') as ff:
        #     self.type_dict = json.load(ff)

        if self.opt.paired:
            self.datalist = [name.strip() for name in self.datalist]
        else:
            self.datalist = [name.strip().split(" ") for name in self.datalist]

        info = f"{opt.datamode} size: {self.__len__()}"
        print(info)
        

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):

        if self.opt.paired:
            img_name = self.datalist[idx]
            return self.fetch_data(img_name)
        else:
            img_name, cloth_name = self.datalist[idx]
            return self.fetch_data(img_name, cloth_name)

    def fetch_data(self, img_name, cloth_name=None):

        if not cloth_name:
            cloth_name = img_name

        # if self.opt.type_mask_condition or self.opt.type_mask_generator:
        #     if int(cloth_name.split(".")[0]) in self.type_dict["Skirts"]:
        #         cloth_type = "skirts"
        #     elif int(cloth_name.split(".")[0]) in self.type_dict["Trousers"]:
        #         cloth_type = "trousers"
        #     else:
        #         raise Exception("A bottom cloth must be either trouser or skirt")

        """ cloth image """
        # read and resize
        path = os.path.join(self.clothroot, 'cloth', cloth_name)
        c = Image.open(path).convert('RGB')
        c = transforms.Resize((self.input_height,self.input_width), interpolation=im.BICUBIC)(c)
        # convert to tensor and normalize
        c = self.transform(c)   # 3*256*192, float32

        """ cloth mask """
        # read and resize
        path = os.path.join(self.clothroot, 'cloth-mask', cloth_name)
        cm = Image.open(path)
        cm = transforms.Resize((self.input_height,self.input_width), interpolation=im.NEAREST)(cm)
        # convert to binary tensor 0 or 1
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32) # [0,1]
        cm = torch.from_numpy(cm_array)     # 256*192, float32
        cm.unsqueeze_(0) # add one more dimension at dim 0, 1*256*192, float32

        """ person image """
        # read and resize
        path = os.path.join(self.imgroot, 'image', img_name)
        p = Image.open(path).convert('RGB')
        p = transforms.Resize((self.input_height,self.input_width), interpolation=im.BICUBIC)(p)
        # convert to tensor and normalize
        p = self.transform(p)       # 3*256*192, float32

        """ person agnostic map """
        # read and resize
        labels = self.get_parse_labels()
        path = os.path.join(self.imgroot, "image-parse-agnostic-v3.2", img_name).replace('jpg','png')
        p_agnostic = Image.open(path)
        p_agnostic = transforms.Resize((self.input_height,self.input_width), interpolation=im.NEAREST)(p_agnostic)
        # convert to Integer tensor
        p_agnostic = torch.from_numpy(np.array(p_agnostic)[None]).long()

        """ agnostic parsing map """
        p_agnostic_parse = torch.FloatTensor(self.opt.parse_map_nc, self.input_height, self.input_width).zero_()
        p_agnostic_parse = p_agnostic_parse.scatter_(0, p_agnostic, 1.0)
        # create a 13-channel map
        new_p_agnostic_parse = torch.FloatTensor(self.semantic_nc, self.input_height, self.input_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_p_agnostic_parse[i] += p_agnostic_parse[label]

        """ warped cloth ground truth """
        if self.opt.target == 'dress':
            cloth_idx = 3
        # elif self.opt.target == 'bottom':
        #     cloth_idx = 4
        else:
            raise Exception("opt.target mush be either 'upper' or 'bottom'")


        warped_msk = 0
        warped_c = 0

        """ dense pose map """
        # read and resize
        path = os.path.join(self.imgroot, "image-densepose", img_name)
        densepose_map = Image.open(path)
        densepose_map = transforms.Resize((self.input_height,self.input_width), interpolation=im.BICUBIC)(densepose_map)
        # convert to tensor and normalize
        densepose_map = self.transform(densepose_map)

        """ agnostic image """
        # read and resize
        path = os.path.join(self.imgroot, "agnostic-v3.2",img_name)
        agnostic = Image.open(path).convert("RGB")
        agnostic = transforms.Resize((self.input_height,self.input_width), interpolation=im.BICUBIC)(agnostic)
        # convert to tensor and normalize
        agnostic = self.transform(agnostic)

        result = {
            "img_name": img_name,   # image name
            "cloth_name": cloth_name,
            # condition input1 (cloth flow)
            "cloth": c,             # cloth in image level (RGB)
            "cloth_mask": cm,       # cloth in mask level (Binary)
            # condition input2 (segmentation network)
            "parse_agnostic": new_p_agnostic_parse,     # clothing-agnostic segmentation map
            "densepose": densepose_map,             # densepose segmentation map
            # generator input
            "agnostic": agnostic,    # clothing agnostic image (RGB)
            # ground truths
            "warped_cloth_mask": warped_msk,       
            "warped_cloth":      warped_c,         
            "image": p                           
        }

        # if self.opt.type_mask_condition or self.opt.type_mask_generator:
        #     if cloth_type == "skirts":
        #         result["cloth_type"] = "skirts"
        #         type_mask = torch.ones_like(cm)
        #     else:
        #         result["cloth_type"] = "trousers"
        #         type_mask = torch.zeros_like(cm)
        #
        # if self.opt.type_mask_condition:
        #     result["parse_agnostic"] = torch.cat([result["parse_agnostic"], type_mask],0)

        return result

    def get_parse_labels(self):

        labels = {
            0: ['background', [0]],
            1: ['hair', [1, 2]],
            2: ['face', [3, 11]],
            3: ['dress', [4, 5, 6, 7, 8]],
            4: ['left_arm', [14]],
            5: ['right_arm', [15]],
            6: ['left_leg', [12]],
            7: ['right_leg', [13]],
            8: ['left_shoe', [9]],
            9: ['right_shoe', [10]],
            10: ['bag', [16]],
            11: ['noise', [17]],
        }
        return labels


class CPDataLoader(object):
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()

        if opt.shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, drop_last=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


if __name__ == "__main__":
    from Configs.Config_test_v1 import ConfigTest
    dataset = HRVTDataset(ConfigTest().get_opt())
    inputs = dataset.__getitem__(5)
    for key in inputs.keys():
        if key != "img_name":
            info = f"{key}:  in shape {list(inputs[key].shape)}"
            print(info)













