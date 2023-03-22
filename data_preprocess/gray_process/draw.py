import numpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import glob
import os
from PIL import Image, ImageDraw
from copy import deepcopy

def parsemask(parse, gray):
    # parse = cv2.cvtColor(numpy.asarray(parse), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(numpy.asarray(gray), cv2.COLOR_RGB2BGR)
    gray_img = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('./test.jpg',gray_img)
    coor = np.where(gray_img < 180)
    parse_cp = deepcopy(parse)
    parse_cp[coor] = 0
    return parse_cp
    # # Binary image
    # ret, thresh_gray = cv2.threshold(gray_img, 250, 255, cv2.THRESH_BINARY)
    #
    # # agnostic + thresh_gray
    # part_black = cv2.bitwise_and(parse, parse, mask=thresh_gray)
    # return  part_black

def close(img):
  kernel = numpy.ones((3, 3), numpy.uint8)
  closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
  return closing

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

def mkdir_if_absent(path):
    if not os.path.exists(path):
        os.makedirs(path)
    elif not os.path.isdir(path):
        # raise FileExistsError
        pass

if __name__ == '__main__':

    parselist = os.listdir('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_0315/model/image-parse-v3')
    parse_image_dir = '/home/vera/Dress_VITON/Dress_master_2501_test/dataset_0315/model/image-parse-v3/'
    gray_dir = '/home/vera/Dress_VITON/Dress_master_2501_test/dataset_0315/model/gray/'

    despath = '/home/vera/Dress_VITON/Dress_master_2501_test/dataset_0315/model/image-parse-agnostic-v3.2'
    mkdir_if_absent(despath)


    for i in range(len(parselist)):
        fullbody_parse = Image.open(os.path.join(parse_image_dir + parselist[i]))
        fullbody_parse.save(os.path.join(despath, parselist[i]))
        fullbody_parse = np.asarray(fullbody_parse)

        gray_name = parselist[i].replace('.png', '.jpg')
        gray_img = Image.open(os.path.join(gray_dir + gray_name))

        parse_res = parsemask(fullbody_parse, gray_img)
        #closing = close(parse_res)
        result = set_palette(parse_res)
        print(i, gray_name)


        result.save(os.path.join(despath, parselist[i]))

        #putpalatte