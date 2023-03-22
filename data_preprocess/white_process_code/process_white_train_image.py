import os
import glob
import cv2
import numpy as np
import base64
from copy import deepcopy
from traceback import format_exc
import requests
from areacal2_0 import filter_process_mask

def mkdir_if_absent(path):
    if not os.path.exists(path):
        os.makedirs(path)
    elif not os.path.isdir(path):
        # raise FileExistsError
        pass

def get_white_parse_img(image,category,upload=0):
    import random
    ARM_CLOTH_PARSE_URL = (
        # 'http://172.28.2.185:8091/cloth_parser/subcategory/',
        # 'http://172.28.2.93:8081/cloth_parser/subcategory/',
        'http://172.28.2.185:8081/cloth_parser/subcategory/',
    )
    st_img = cv2.imencode('.jpg', image)[1].tostring()
    data = {
        'image_data': base64.b64encode(st_img).decode('UTF-8'),
        'category': category,
        'upload': upload,
        'is_white':True
    }

    res = requests.post(
        random.choice(ARM_CLOTH_PARSE_URL), data=data, timeout=60,headers={'content-type': 'application/x-www-form-urlencoded'},)
    result = res.json()
    return result

def extend_white_image_v3(image,bbox,mask,top_center_ratio,min_edge_top_ratio=0.07,min_edge_w_ratio=0.15,min_edge_h_ratio=0.15):
    h_w_ratio = 4 / 3
    h,w = image.shape[:2]
    b_h, b_w = bbox[3] - bbox[1],bbox[2] - bbox[0]
    #### normal
    # min_edge_w_ratio, min_edge_h_ratio = 0.15, 0.15
    #### long sleeve
    # min_edge_w_ratio, min_edge_h_ratio = 0.13, 0.13
    min_w = round(b_w + b_w * min_edge_w_ratio*2)
    min_h = round(b_h + b_h * min_edge_h_ratio * 2)

    if min_w * h_w_ratio > min_h:
        des_h = round(min_w * h_w_ratio)
        des_w = min_w
    else:
        des_h = min_h
        des_w = round(min_h/h_w_ratio)
    #### normal
    # min_edge_top_ratio = 0.07
    #### long sleeve
    # min_edge_top_ratio = 0.06
    final_top_center_ratio = max(min_edge_top_ratio+b_h/(des_h*2),top_center_ratio)
    left_extra_width = int(round((des_w - b_w) / 2))
    right_extra_width = des_w - b_w - left_extra_width
    top_extra_height = int(round((des_h*final_top_center_ratio - b_h/2)))
    bottom_extra_height = des_h - b_h - top_extra_height
    extend_bbox = [bbox[0]-left_extra_width,bbox[1]-top_extra_height,bbox[2]+right_extra_width,bbox[3]+bottom_extra_height]
    raw_bbox = [max(0,extend_bbox[0]),max(0,extend_bbox[1]),min(w,extend_bbox[2]),min(h,extend_bbox[3])]

    corner_pix = 5
    avg_colors = list()
    for h1, h2, w1, w2 in [[0, corner_pix, 0, corner_pix], [-corner_pix, -1, 0, corner_pix],
                           [0, corner_pix, -corner_pix, -1], [-corner_pix, -1, -corner_pix, -1]]:
        left_corner = image[h1:h2, w1:w2]
        avg_color = np.average(left_corner, axis=(0, 1))
        avg_colors.append(avg_color)

    avg_colors = np.asarray(avg_colors)
    avg_color = np.average(avg_colors,axis=0)

    image = image[raw_bbox[1]:raw_bbox[3], raw_bbox[0]:raw_bbox[2]]
    mask = mask[raw_bbox[1]:raw_bbox[3], raw_bbox[0]:raw_bbox[2]]
    copy_bbox = [value - raw_bbox[i] for i, value in enumerate(extend_bbox)]
    copy_bbox[0] = -copy_bbox[0]
    copy_bbox[1] = -copy_bbox[1]

    max_pix = 3
    if abs(raw_bbox[0] - bbox[0]) < max_pix or abs(raw_bbox[1] - bbox[1]) < max_pix or \
            abs(raw_bbox[2] - bbox[2]) < max_pix or abs(raw_bbox[3] - bbox[3]) < max_pix:
        des_image = cv2.copyMakeBorder(image, copy_bbox[1], copy_bbox[3], copy_bbox[0], copy_bbox[2],
                                       cv2.BORDER_CONSTANT, value=tuple(avg_color))
    else:
        des_image = cv2.copyMakeBorder(image, copy_bbox[1], copy_bbox[3], copy_bbox[0], copy_bbox[2],
                                       cv2.BORDER_REPLICATE)

    des_mask = cv2.copyMakeBorder(mask, copy_bbox[1], copy_bbox[3], copy_bbox[0], copy_bbox[2],
                                       cv2.BORDER_REPLICATE)
    return des_image, des_mask


def process_bottom_cloth():
    src_image_dir = '/home/vera/Dress_VITON/Dress_master_2501_test/dataset_0306/dress/pattern/color/cloth/*.jpg'
    des_base_dir = '/home/vera/Dress_VITON/Dress_master_2501_test/dataset_0306/dress/pattern/color'
    mask_origin_dir = os.path.join(des_base_dir,'white_2w_origin_mask')
    image_dir = os.path.join(des_base_dir, 'white_2w_cloth')
    mask_dir = os.path.join(des_base_dir, 'white_2w_cloth-mask')
    large_resize_dir = os.path.join(des_base_dir,'white_2w_large_resize')
    mkdir_if_absent(mask_origin_dir)
    mkdir_if_absent(image_dir)
    mkdir_if_absent(mask_dir)
    mkdir_if_absent(large_resize_dir)
    e_h,e_w = 1024,768
    max_h = 2500
    for image_path in glob.glob(src_image_dir):
        try:
            print(image_path)
            name = image_path.strip().split('/')[-1]
            if os.path.exists(os.path.join(image_dir, name)):
                print('exist')
                continue
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            if h > max_h:
                image = cv2.resize(image,(round(w*max_h/h),max_h))
                cv2.imwrite(os.path.join(large_resize_dir,name),image)
                print('resize',h,w,image.shape[0],image.shape[1])
                h,w = image.shape[:2]

            resp_json = get_white_parse_img(image, category=60, upload=0)
            if resp_json.get('ret') == 0:
                parsed_raw = list(resp_json.get('data').values())[0]
                im_binary = np.fromstring(base64.b64decode(parsed_raw), np.uint8)

                parsed_image = cv2.imdecode(im_binary, -1)
                ori_mask = deepcopy(parsed_image[:, :, 3].reshape(h, w, 1))
                is_valid, mask = filter_process_mask(ori_mask)
                if not is_valid:
                    print('mask not valid', name)
                    # cv2.imwrite(
                    #     os.path.join(invalid_mask_des_dir, '{pid}_{image_id}.jpg'.format(pid=pid, image_id=image_id)), mask)
                    continue
                cv2.imwrite(os.path.join(mask_origin_dir,name),ori_mask)
                mask_cp = deepcopy(mask)
                mask_cp = cv2.erode(mask_cp, np.ones((4, 4), np.uint8), iterations=1)
                coor = np.where(mask_cp > 10)
                min_x, max_x = min(coor[1]), max(coor[1])
                min_y, max_y = min(coor[0]), max(coor[0])
                bbox = [min_x, min_y, max_x, max_y]
                top_center_ratio = 1 / 2
                des_image, des_mask = extend_white_image_v3(image, bbox, mask, top_center_ratio, min_edge_top_ratio=0.07,
                                                            min_edge_w_ratio=0.15, min_edge_h_ratio=0.15)
                cv2.imwrite(os.path.join(image_dir, name), cv2.resize(des_image, (e_w, e_h), cv2.INTER_AREA))
                cv2.imwrite(os.path.join(mask_dir, name), cv2.resize(des_mask, (e_w, e_h), cv2.INTER_AREA))
        except:
            print(format_exc())

if __name__ == '__main__':
    process_bottom_cloth()
    # src_image_dir = '/home/vera/Dress_VITON/Dress_master_2501_test/mid_dress_dataset/cloth/*.jpg'
    # des_dir = '/home/vera/Dress_VITON/Dress_master_2501_test/mid_dress_dataset/cloth-mask'
    # for image_path in glob.glob(src_image_dir):
    #
    #     # print(image_path)
    #     name = image_path.strip().split('/')[-1]
    #     image = cv2.imread(image_path)
    #     new_img = get_white_parse_img(image, category=61, upload=0 )
    #     cv2.imwrite(os.path.join(des_dir, name), new_img)
    #
    # print('done')

    # from PIL import Image
    # image = Image.open('/home/vera/copy_HR/Dress_VTON/data/cloth-mask/141220.jpg')
    # image_array = np.asarray(image)