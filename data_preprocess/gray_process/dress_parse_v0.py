#coding=utf-8
import os

import numpy
# import torch
# import torch.utils.data as data
# import torchvision.transforms as transforms

from PIL import Image, ImageDraw
import json
import cv2
import os.path as osp
import numpy as np
from copy import deepcopy


def get_bottom_agnostic(im, im_parse, pose_data):
        #each part of human body
    parse_array = np.array(im_parse)
    parse_head = ((parse_array == 2).astype(np.float32) +
                (parse_array == 11).astype(np.float32))

    parse_shoes = ((parse_array == 9).astype(np.float32)+
                (parse_array == 10).astype(np.float32))
    parse_arm = ((parse_array == 14).astype(np.float32)+
                (parse_array == 15).astype(np.float32))

    parse_upper = (parse_array == 4).astype(np.float32)

    #draw agnostic image on the original image
    agnostic = im.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    gray = Image.new("RGB", (768 , 1024), (255, 255, 255))
    gray_draw = ImageDraw.Draw(gray)

    white = gray.copy()

    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    length_b = np.linalg.norm(pose_data[12] - pose_data[9])
    point = (pose_data[9] + pose_data[12]) / 2

    pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
    pose_data[12] = point + (pose_data[12] - point) / length_b * length_a

    r = int(length_a / 16) + 1

    #mask top torso
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

    gray_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r * 6)
    gray_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r * 6)
    gray_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r * 12)
    gray_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

    # mask neck
    pointx, pointy = pose_data[1]
    agnostic_draw.rectangle((pointx - r * 5, pointy - r * 9, pointx + r * 5, pointy), 'gray', 'gray')

    gray_draw.rectangle((pointx - r * 5, pointy - r * 9, pointx + r * 5, pointy), 'gray', 'gray')

    # mask arms
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r * 12)
    gray_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r * 12)

    for i in [2, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx - r * 5, pointy - r * 6, pointx + r * 5, pointy + r * 6), 'gray', 'gray')
        gray_draw.ellipse((pointx - r * 5, pointy - r * 6, pointx + r * 5, pointy + r * 6), 'gray', 'gray')

    for i in [3, 4, 6, 7]:
        if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and
                                                                           pose_data[i, 1] == 0.0):
            continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), 'gray', 'gray')
        gray_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), 'gray', 'gray')

    for parse_id, pose_ids in [(14, [5, 6, 7]), (15, [2, 3, 4])]:
        # mask_arm = Image.new('L', (self.fine_width, self.fine_height), 'white')
        mask_arm = Image.new('L', (768, 1024), 'white')
        mask_arm_draw = ImageDraw.Draw(mask_arm)
        pointx, pointy = pose_data[pose_ids[0]]
        mask_arm_draw.ellipse((pointx - r * 5, pointy - r * 6, pointx + r * 5, pointy + r * 6), 'black', 'black')
        for i in pose_ids[1:]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (
                    pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            mask_arm_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'black', width=r * 10)
            pointx, pointy = pose_data[i]
            if i != pose_ids[-1]:
                mask_arm_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), 'black',
                                      'black')
        mask_arm_draw.ellipse((pointx - r * 4, pointy - r * 4, pointx + r * 4, pointy + r * 4), 'black', 'black')

        parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))
        gray.paste(im, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))


    #mask bottom
    for i in [9, 12]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx - r * 8, pointy - r * 6, pointx + r * 8, pointy + r * 6), 'gray', 'gray')
        gray_draw.ellipse((pointx - r * 8, pointy - r * 6, pointx + r * 8, pointy + r * 6), 'gray', 'gray')

        agnostic_draw.rectangle((pointx - r * 10, pointy - r * 10, pointx + r * 10, pointy + r * 10), 'gray', 'gray')
        gray_draw.rectangle((pointx - r * 10, pointy - r * 10, pointx + r * 10, pointy + r * 10), 'gray', 'gray')
    for i in [10, 13]:
        pointx, pointy = pose_data[i]
        agnostic_draw.rectangle((pointx - r * 10, pointy - r * 7, pointx + r * 10, pointy + r * 7), 'gray', 'gray')

        gray_draw.rectangle((pointx - r * 10, pointy - r * 7, pointx + r * 10, pointy + r * 7), 'gray', 'gray')

    for i in [11, 14]:
        pointx, pointy = pose_data[i]
        agnostic_draw.rectangle((pointx - r * 9, pointy - r * 8, pointx + r * 9, pointy + r * 8), 'gray', 'gray')

        gray_draw.rectangle((pointx - r * 9, pointy - r * 8, pointx + r * 9, pointy + r * 8), 'gray', 'gray')
    agnostic_draw.line([tuple(pose_data[i]) for i in [9, 10]], 'gray', width=r * 20)
    agnostic_draw.line([tuple(pose_data[i]) for i in [12, 13]], 'gray', width=r * 20)
    agnostic_draw.line([tuple(pose_data[i]) for i in [10, 11]], 'gray', width=r * 20)
    agnostic_draw.line([tuple(pose_data[i]) for i in [13, 14]], 'gray', width=r * 20)
    agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r * 30)
    agnostic_draw.line([tuple(pose_data[i]) for i in [10, 13]], 'gray', width=r * 12)
    agnostic_draw.line([tuple(pose_data[i]) for i in [11, 14]], 'gray', width=r * 15)
    agnostic_draw.line([tuple(pose_data[i]) for i in [9, 11]], 'gray', width=r * 20)
    agnostic_draw.line([tuple(pose_data[i]) for i in [12, 14]], 'gray', width=r * 20)

    gray_draw.line([tuple(pose_data[i]) for i in [9, 10]], 'gray', width=r * 20)
    gray_draw.line([tuple(pose_data[i]) for i in [12, 13]], 'gray', width=r * 20)
    gray_draw.line([tuple(pose_data[i]) for i in [10, 11]], 'gray', width=r * 20)
    gray_draw.line([tuple(pose_data[i]) for i in [13, 14]], 'gray', width=r * 20)
    gray_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r * 30)
    gray_draw.line([tuple(pose_data[i]) for i in [10, 13]], 'gray', width=r * 12)
    gray_draw.line([tuple(pose_data[i]) for i in [11, 14]], 'gray', width=r * 15)
    gray_draw.line([tuple(pose_data[i]) for i in [9, 11]], 'gray', width=r * 20)
    gray_draw.line([tuple(pose_data[i]) for i in [12, 14]], 'gray', width=r * 20)
    # agnostic_draw.line([tuple(pose_data[i]) for i in [11, 14]], 'gray', width=r * 8)
    pointx, pointy = pose_data[8]
    agnostic_draw.rectangle((pointx - r * 10, pointy - r * 10, pointx + r * 10, pointy + r * 10), 'gray', 'gray')

    agnostic_draw.polygon([tuple(pose_data[i]) for i in [9, 12, 13, 10]], 'gray', 'gray')
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [10, 13, 14, 11]], 'gray', 'gray')

    gray_draw.rectangle((pointx - r * 10, pointy - r * 10, pointx + r * 10, pointy + r * 10), 'gray', 'gray')

    gray_draw.polygon([tuple(pose_data[i]) for i in [9, 12, 13, 10]], 'gray', 'gray')
    gray_draw.polygon([tuple(pose_data[i]) for i in [10, 13, 14, 11]], 'gray', 'gray')

    agnostic.paste(im, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))
    agnostic.paste(im, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    agnostic.paste(im, None, Image.fromarray(np.uint8(parse_shoes * 255), 'L'))

    gray.paste(white, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))
    gray.paste(white, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    gray.paste(white, None, Image.fromarray(np.uint8(parse_shoes * 255), 'L'))

    return agnostic, gray

def openpose_res(openpose_name):
    with open(osp.join(openpose_json_dir, openpose_name),'r') as f:
        pose_label = json.load(f)
        pose_data = pose_label['people'][0]['pose_keypoints_2d']
        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1, 3))[:, :2]
    return pose_data


def combine(agnostic, gray):
    # PIL-> cv2 RGB ->BGR
    # gray -> binary mask gray
    agnostic_copy = agnostic.copy()
    agnostic_copy = cv2.cvtColor(numpy.asarray(agnostic_copy), cv2.COLOR_RGB2BGR)

    gray_img = gray.copy()
    gray_img = cv2.cvtColor(numpy.asarray(gray_img), cv2.COLOR_RGB2BGR)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    # Binary image
    ret, thresh_gray = cv2.threshold(gray_img, 250, 255, cv2.THRESH_BINARY)

    # agnostic + thresh_gray
    part_black = cv2.bitwise_and(agnostic_copy, agnostic_copy, mask=thresh_gray)

    return part_black
#
def combine_png(origin, gray):
    gray_img = gray.copy()
    gray_img = cv2.cvtColor(numpy.asarray(gray_img), cv2.COLOR_RGB2BGR)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    # Binary image
    ret, thresh_gray = cv2.threshold(gray_img, 250, 255, cv2.THRESH_BINARY)

    # agnostic + thresh_gray
    part_black = cv2.bitwise_and(origin, origin, mask=thresh_gray)

    return part_black

def get_trans(part_black):
    tmp = cv2.cvtColor(part_black, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(part_black)
    rgba = [b, g, r, alpha]
    transparency = cv2.merge(rgba, 4)

    return transparency

def parsemask(parse, gray):

    gray = cv2.cvtColor(numpy.asarray(gray), cv2.COLOR_RGB2BGR)
    gray_img = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    coor = np.where(gray_img < 180)
    parse_cp = deepcopy(parse)
    parse_cp[coor] = 0
    return parse_cp

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def set_palette(image):
    output_img = Image.fromarray(np.asarray(image, dtype=np.uint8))
    palette = get_palette(18)
    output_img.putpalette(palette)
    return output_img




if __name__ == '__main__':
    imagelist = os.listdir('dataset_resize1024/person_image')
    person_image_dir = 'dataset_resize1024/person_image/'
    parse_dir = 'dataset_resize1024/image-parse-v3/'
    openpose_img_dir = 'dataset_resize1024/openpose_img/'
    openpose_json_dir = 'dataset_resize1024/openpose_json/'

    despath = 'bottom_output/fullbody_output'
    des_gray_path = 'bottom_output/gray'
    transparency_path = 'demo_transparency'

    for i in range(len(imagelist)):
        im_pil_big = Image.open(person_image_dir + imagelist[i])
        parse_name = imagelist[i].replace('.jpg', '.png')
        im_parse_pil_big = Image.open(parse_dir + parse_name)

        openpose_name = imagelist[i].replace('.jpg', '_keypoints.json')
        op_res = openpose_res(openpose_name)

        agnostic, gray = get_bottom_agnostic(im_pil_big, im_parse_pil_big, op_res)

        # com_img = combine_png(im_parse_pil_big, gray)
        # transparency_img = get_trans(com_img)

        # img save
        agnostic.save(os.path.join(despath, imagelist[i]))
        gray.save(os.path.join(des_gray_path, imagelist[i]))
