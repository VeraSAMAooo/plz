import os
import torch
import json
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode as im
from PIL import Image
import numpy as np
import collections

DIRTY_LIST = ["8493539.jpg","7084462.jpg","9097034.jpg","10286153.jpg","8078712.jpg", "9604694.jpg", "10255340.jpg", "7167147.jpg",
              "9626050.jpg","8713530.jpg",]

class HRVTDataset(Dataset):
    """ Dataset for HR-VITON"""
    def __init__(self, opt, for_generator=False):
        super().__init__()
        # base setting
        self.for_generator = for_generator
        self.opt = opt
        self.datamode = opt.datamode    # train or test or self-defined
        self.root = opt.dataroot
        
        if for_generator:
            self.fine_height = opt.input_height
            self.fine_width = opt.input_width
        else:
            self.fine_height = opt.fine_height
            self.fine_width = opt.fine_width

        self.semantic_nc = opt.semantic_nc
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                # normalize each channel with mean = 0.5 and std = 0.5
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.datalist = opt.train_list if self.datamode=="train" else opt.test_list
        with open(os.path.join(self.root,self.datalist),"r") as ff:
            self.datalist = ff.readlines()
            ff.close()

        # with open(opt.type_dict_path,'r') as ff:
        #     self.type_dict = json.load(ff)

        # if self.opt.paired:
        #     self.datalist = [name.strip() for name in self.datalist]
        #     if opt.sample and self.datamode == "train":
        #         self.datalist = self.sample_datalist(self.datalist,opt)
        # else:
        
        if opt.sample:
            self.datalist = self.sample_datalist(self.datalist,opt)
        self.datalist = [name.strip().split(" ") for name in self.datalist]
        self.datalist = [file for file in self.datalist if file[0] not in DIRTY_LIST]

        info = f"{opt.datamode} size: {self.__len__()}"
        print(info)

        # crop image
        # crop_factor = 0.2
        # top, left = int(1024*crop_factor), int(768*crop_factor/2)
        # height, width = 1024-top, 768-2*left
        # self.crop_size = (top,left,height,width)
            

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):

        if len(self.datalist[idx]) < 2:
            img_name = self.datalist[idx][0]
            cloth_name = img_name
        else:
            img_name, cloth_name = self.datalist[idx]
        inputs_paired = self.fetch_data(img_name)
        if not self.for_generator:
            inputs_unpaired = self.fetch_data(img_name, cloth_name)
            return inputs_paired, inputs_unpaired
        else: return inputs_paired


    def sample_datalist(self, datalist, opt):

        assert opt.paired is True, "datalist can only be sampled in pired mode"

        sample_rate = opt.sample_rate

        type_dict = self.type_dict

        types = opt.sample_types
        
        data_from_type = collections.defaultdict(list)

        for data in datalist:
            for type_name in types:
                if int(data.split(".")[0]) in type_dict[type_name]:
                    data_from_type[type_name].append(data)
        
        new_datalist = []
        for type_list in data_from_type.values():
            sample_num = int(sample_rate * len(type_list))
            new_datalist += list(np.random.choice(type_list,sample_num))

        return new_datalist
            

    def fetch_data(self, img_name, cloth_name=None):

        if not cloth_name:
            cloth_name = img_name

        # if self.opt.type_mask:
        #     if int(cloth_name.split(".")[0]) in self.type_dict["Skirts"]:
        #         cloth_type = "skirts"
        #     elif int(cloth_name.split(".")[0]) in self.type_dict["Trousers"]:
        #         cloth_type = "trousers"
        #     else:
        #         raise Exception("A bottom cloth must be either trouser or skirt")

        """ cloth image """
        # read and resize
        path = os.path.join(self.root, 'cloth', cloth_name)
        c = Image.open(path).convert('RGB')
        c = transforms.Resize(self.fine_width, interpolation=im.BICUBIC)(c)
        # convert to tensor and normalize
        c = self.transform(c)   # 3*256*192, float32

        """ cloth mask """
        # read and resize
        path = os.path.join(self.root, 'cloth-mask', cloth_name)
        cm = Image.open(path)
        cm = transforms.Resize(self.fine_width, interpolation=im.NEAREST)(cm)
        # convert to binary tensor 0 or 1
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32) # [0,1]
        cm = torch.from_numpy(cm_array)     # 256*192, float32
        cm.unsqueeze_(0) # add one more dimension at dim 0, 1*256*192, float32

        """ person image """
        # read and resize
        path = os.path.join(self.root, 'image', img_name)
        p = Image.open(path).convert('RGB')
        p = transforms.Resize(self.fine_width, interpolation=im.BICUBIC)(p)
        # convert to tensor and normalize
        p = self.transform(p)       # 3*256*192, float32

        labels = self.get_parse_labels()
        """ person parsing image """
        # read and resize
        if not self.for_generator or self.opt.train_with_GT:
            path = os.path.join(self.root, 'image-parse-v3', img_name).replace("jpg", "png")
            p_parse = Image.open(path)
            p_parse = transforms.Resize(self.fine_width, interpolation=im.NEAREST)(p_parse)
            # convert to integer tensor
            p_parse = torch.from_numpy(np.array(p_parse)[None]).long() # add one more dim at 0 using array[None]

            """ parse map """
            # labels for parsing
            # broadcast each integer to corresponding channel
            parse_map = torch.FloatTensor(self.opt.parse_map_nc, self.fine_height, self.fine_width).zero_()
            parse_map = parse_map.scatter_(0, p_parse, 1.0)
            # convert to 13-channel parsing map
            new_parse_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()
            for i in range(len(labels)):
                for label in labels[i][1]:
                    new_parse_map[i] += parse_map[label]
            # compress 13 -channel parsing map to 1-channel parsing map
            parse_onehot = torch.FloatTensor(1, self.fine_height, self.fine_width).zero_()
            for i in range(len(labels)):
                for label in labels[i][1]:
                    parse_onehot[0] += parse_map[label] * i
        else:
            path = os.path.join(self.root, 'fake_segmap', img_name).replace("jpg", "pt")
            new_parse_map = torch.load(path)[0]
            parse_onehot = "none"


        """ person agnostic map """
        # read and resize
        path = os.path.join(self.root, "image-parse-agnostic-v3.2", img_name).replace("jpg", "png")
        p_agnostic = Image.open(path)
        p_agnostic = transforms.Resize(self.fine_width, interpolation=im.NEAREST)(p_agnostic)
        # convert to Integer tensor
        p_agnostic = torch.from_numpy(np.array(p_agnostic)[None]).long()

        """ agnostic parsing map """
        p_agnostic_parse = torch.FloatTensor(self.opt.parse_map_nc, self.fine_height, self.fine_width).zero_()
        p_agnostic_parse = p_agnostic_parse.scatter_(0, p_agnostic, 1.0)
        # create a 13-channel map
        new_p_agnostic_parse = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()
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
        if not self.for_generator or self.opt.train_with_GT:
            warped_msk = new_parse_map[cloth_idx:cloth_idx+1]
            warped_c = p * warped_msk + (1 - warped_msk)
        else:
            # warped cloth
            path = os.path.join(self.root, 'fake_warped_cloth', cloth_name).replace("jpg","pt")
            warped_c = torch.load(path)[0]
            # warped_c = Image.open(path).convert('RGB')
            # warped_c = transforms.Resize(self.fine_width, interpolation=im.BICUBIC)(warped_c)
            # # convert to tensor and normalize
            # warped_c = self.transform(warped_c)       # 3*256*192, float32

            # # warped cloth mask
            path = os.path.join(self.root, 'fake_warped_clothmask', cloth_name).replace("jpg","pt")
            warped_msk = torch.load(path)[0]
            # warped_msk = Image.open(path)
            # warped_msk = transforms.Resize(self.fine_width, interpolation=im.NEAREST)(warped_msk)
            # # convert to binary tensor 0 or 1
            # cm_array = np.array(warped_msk)
            # cm_array = (cm_array >= 128).astype(np.float32) # [0,1]
            # warped_msk = torch.from_numpy(cm_array)     # 256*192, float32
            # warped_msk.unsqueeze_(0) # add one more dimension at dim 0, 1*256*192, float32
            

        """ dense pose map """
        # read and resize
        path = os.path.join(self.root, "image-densepose", img_name)
        densepose_map = Image.open(path)
        densepose_map = transforms.Resize(self.fine_width, interpolation=im.BICUBIC)(densepose_map)
        # convert to tensor and normalize
        densepose_map = self.transform(densepose_map)

        """ agnostic image """
        # read and resize
        path = os.path.join(self.root, "agnostic-v3.2",img_name)
        agnostic = Image.open(path).convert("RGB")
        agnostic = transforms.Resize(self.fine_width, interpolation=2)(agnostic)
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
            "parse_onehot": parse_onehot,          # CE loss
            "parse": new_parse_map,                # GAN loss
            "warped_cloth_mask": warped_msk,       # L1 loss
            "warped_cloth":      warped_c,         # VGG loss
            "image": p                             # Final GT
        }

        # if self.opt.type_mask:
        #     result["cloth_type"] = cloth_type
        #     if cloth_type == "skirts":
        #         type_mask = torch.ones_like(cm)
        #     else:
        #         type_mask = torch.zeros_like(cm)
        #     result["parse_agnostic"] = torch.cat([result["parse_agnostic"], type_mask],0)

        # if self.for_generator:
        #     for key, val in result.items():
        #         if key == "img_name" or key == "cloth_name" or key == "cloth_type":
        #             continue
        #         # val = transforms.functional.crop(val,*self.crop_size)
        #         val = transforms.functional.resize(val, (self.opt.fine_height, self.opt.fine_width))
        #         result[key] = val

        return result

    def get_parse_labels(self):

        labels = {
            0: ['background',  [0]],
            1: ['hair',        [1, 2]],
            2: ['face',        [3, 11]],
            3: ['dress',       [4, 5, 6, 7, 8]],
            4: ['left_arm',    [14]],
            5: ['right_arm',   [15]],
            6: ['left_leg',    [12]],
            7: ['right_leg',   [13]],
            8: ['left_shoe',   [9]],
            9: ['right_shoe',  [10]],
            10: ['bag',        [16]],
            11: ['noise',      [17]],
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
    from Configs.Config_condition_v1 import Config
    dataset = HRVTDataset(Config().get_opt())
    inputs = dataset.__getitem__(5)
    for key in inputs.keys():
        if key != "img_name" and inputs[key]:
            info = f"{key}:  in shape {list(inputs[key].shape)}"
            print(info)













