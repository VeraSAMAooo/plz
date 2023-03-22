import os
from PIL import Image, ImageDraw
import json
import cv2
import os.path as osp
import numpy as np
import math
import logging
#read file

logging.basicConfig(filename="test.log",
                    format="%(asctime)-15s %(levelname)s %(message)s",
                    filemode='w',
                    level=logging.DEBUG,
                    )
logger = logging.getLogger()

path = '/home/vera/myCode/top_pose_process_examples/set_0309/image' # Source Folder
dstpath = '/home/vera/myCode/top_pose_process_examples/set_0309/image'

if not os.path.exists(dstpath):   # create path if the path is not exit
    os.makedirs(dstpath)

files = os.listdir(path)

for image in files:
    try:
        img = cv2.imread(os.path.join(path,image), cv2.IMREAD_UNCHANGED)
        h, w, _ = img.shape
        print("h,w :",h, w)
        des_w = round((h * 3) / 4)
        add_w = int((des_w - w) / 2)
        #print(image, add)
        if add_w >= 0:
            new_img = cv2.copyMakeBorder(img, 0, 0, add_w, des_w-w-add_w, cv2.BORDER_CONSTANT, value = (255, 255, 255))
            newnew_img = cv2.resize(new_img,(768,1024))
            hh, ww, _ = newnew_img.shape
            print("hh,ww :", hh, ww)
            cv2.imwrite(os.path.join(dstpath, image), newnew_img)
        #
        #
        # else:
        #     new_img = img[:,(-1)*add:(-1)*add+des_w]
        #     new_img = cv2.copyMakeBorder(img, 0, 0, (-1)*add, (-1)*add, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        #     newnew_img = cv2.resize(new_img, (768, 1024))
        #     hh, ww, _ = newnew_img.shape
        #     print("hh,ww :", hh, ww)
        #     cv2.imwrite(os.path.join(dstpath, image), newnew_img)
        else:
            des_h = round((w * 4) / 3)
            add_h = int((des_h - h) / 2)
            new_img = cv2.copyMakeBorder(img, add_h, des_h - h - add_h, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            newnew_img = cv2.resize(new_img, (768, 1024))
            hh, ww, _ = newnew_img.shape

            cv2.imwrite(os.path.join(dstpath, image), newnew_img)

        dic_true = {
            'img name: ', image
        }
        logging.log(level=logging.DEBUG, msg=dic_true, exc_info=True)
    except:
        dic_false = {
            'something wrong and img name: ': image
        }
        print('something strange wowo')
        logging.log(level=logging.DEBUG, msg=dic_false, exc_info=True)





