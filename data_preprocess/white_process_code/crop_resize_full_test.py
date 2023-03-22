import os.path as osp
import shutil
import math
import os
import glob
import json
import numpy as np
import cv2
from traceback import format_exc
from PIL import Image

def mkdir_if_absent(path):
    if not os.path.exists(path):
        os.makedirs(path)
    elif not os.path.isdir(path):
        # raise FileExistsError
        pass

def get_valid_parse_and_bbox():

    pid_bbox_map = dict()
    count = 0
    valid_parse_pids = list()
    for image_path in glob.glob('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/image-parse-v3/*.png'):
        try:
            count += 1
            print(count,image_path)
            name = image_path.strip().split('/')[-1]
            pid = int(name.split('.')[0])

            image_pil = Image.open(image_path)
            w,h = image_pil.size

            img_array = np.asarray(image_pil)
            parse_lower = ((img_array == 7).astype(np.float32) )
            # print('bottom_len',len(np.where(parse_lower > 0)[0]))
            coor = np.where(img_array > 0)
            if len(coor[0]) <= 0 or len(np.where(parse_lower > 0)[0])/len(coor[0]) <= 0.05:
                print('no bottom')
                # image_pil.save(os.path.join(des_base_dir, name.replace('.png','_no_botoom.png')))
                continue

            min_y,max_y = min(coor[0]),max(coor[0])
            min_x, max_x = min(coor[1]), max(coor[1])
            full_bbox = [int(min_x),int(min_y),int(max_x),int(max_y)]
            valid_parse_pids.append(pid)
            pid_bbox_map[pid] = full_bbox

            # count += 1
            # if count > 100:
            #     break
        except:
            print(format_exc())

    print(len(valid_parse_pids),valid_parse_pids[:10])

    with open('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/fullbody_parse_2w_valid_pids.json','w') as file:
        file.write(json.dumps(valid_parse_pids))

    with open('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/fullbody_parse_2w_valid_pid_bbox_map.json','w') as file:
        file.write(json.dumps(pid_bbox_map))


def resize_bottom_train_full_image():

    with open('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/fullbody_parse_2w_valid_pid_bbox_map.json','r') as file:
        pid_bbox_map = json.loads(file.read())
    print(len(pid_bbox_map))

    with open('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/fullbody_parse_2w_valid_pids.json','r') as file:
        valid_pids = json.loads(file.read())
    print(len(pid_bbox_map))

    des_base_dir = '/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/image_v1'
    mkdir_if_absent(des_base_dir)

    ratio = 3/4
    e_h,e_w = 1024,768

    pid_crop_pad_bbox = dict()
    for idx,image_path in enumerate(glob.glob('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/image/*.jpg')):
        try:
            print(idx,image_path)
            name = image_path.strip().split('/')[-1]
            pid = int(name.split('.')[0])
            if pid not in valid_pids:
                print('not valid',image_path)
                continue
            full_bbox = pid_bbox_map.get(str(pid))
            image = cv2.imread(image_path)
            h,w = image.shape[:2]

            y0,y1 = full_bbox[1],full_bbox[3]
            expand_y0, expand_y1 = max(0,round(y0-(y1-y0)*0.05)),min(h,round(y1+(y1-y0)*0.05))
            expand_h = expand_y1 - expand_y0
            expand_w = round(expand_h*ratio)
            x0,x1 = full_bbox[0],full_bbox[2]
            ideal_expand_x0,ideal_expand_x1 = round((x1+x0-expand_w)/2),round((x1+x0-expand_w)/2)+expand_w
            expand_x0, expand_x1 = max(0,ideal_expand_x0),min(w,ideal_expand_x1)
            crop_bbox = [expand_x0,expand_y0,expand_x1,expand_y1]
            pad_bbox = [expand_x0 - ideal_expand_x0,0,ideal_expand_x1-expand_x1,0]
            print(crop_bbox,pad_bbox)
            crop_image = image[crop_bbox[1]:crop_bbox[3],crop_bbox[0]:crop_bbox[2]]
            des_image = cv2.copyMakeBorder(crop_image, pad_bbox[1], pad_bbox[3], pad_bbox[0], pad_bbox[2],
                                           cv2.BORDER_REPLICATE)
            des_h,des_w = des_image.shape[:2]
            if(des_h > e_h):
                cv2.imwrite(os.path.join(des_base_dir,name),cv2.resize(des_image,(e_w,e_h),cv2.INTER_AREA))
            else:
                cv2.imwrite(os.path.join(des_base_dir, name), cv2.resize(des_image, (e_w, e_h), cv2.INTER_CUBIC))
            pid_crop_pad_bbox[pid] = {'crop_bbox':crop_bbox,'pad_bbox':pad_bbox}
            # cv2.imwrite(os.path.join(des_base_dir, name.replace('.jpg','_origin.jpg')), image)
            # if idx > 100:
            #     break
        except:
            print(format_exc())

    with open('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/pid_crop_pad_bbox.json','w') as file:
        file.write(json.dumps(pid_crop_pad_bbox))

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


def resize_bottom_train_parse_image():

    with open('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/fullbody_parse_2w_valid_pids.json','r') as file:
        valid_pids = json.loads(file.read())

    with open('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/pid_crop_pad_bbox.json','r') as file:
        pid_crop_pad_bbox = json.loads(file.read())

    des_base_dir = '/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/parse_v1'
    mkdir_if_absent(des_base_dir)

    ratio = 3 / 4
    e_h, e_w = 1024, 768

    for idx, image_path in enumerate(
            glob.glob('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/image-parse-v3/*.png')):
        try:
            print(idx, image_path)
            name = image_path.strip().split('/')[-1]
            pid = int(name.split('.')[0])
            if pid not in valid_pids:
                print('not valid', image_path)
                continue
            if os.path.exists(os.path.join(des_base_dir,name)):
                print('exist')
                continue
            crop_pad_bbox = pid_crop_pad_bbox.get(str(pid))
            image_pil = Image.open(image_path)
            w, h = image_pil.size

            img_array = np.asarray(image_pil)
            crop_bbox = crop_pad_bbox.get('crop_bbox')
            pad_bbox = crop_pad_bbox.get('pad_bbox')
            crop_image = img_array[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]]
            des_image = cv2.copyMakeBorder(crop_image, pad_bbox[1], pad_bbox[3], pad_bbox[0], pad_bbox[2],
                                           cv2.BORDER_REPLICATE)

            # print(crop_bbox, pad_bbox)
            output_img = Image.fromarray(np.asarray(des_image, dtype=np.uint8))
            output_img = output_img.resize((e_w,e_h),Image.Resampling.NEAREST)
            palette = get_palette(18)
            output_img.putpalette(palette)
            output_img.save(os.path.join(des_base_dir,name))
            # if idx > 100:
            #     break
        except:
            print(format_exc())

def resize_bottom_train_pose_image():

    with open('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/fullbody_parse_2w_valid_pids.json','r') as file:
        valid_pids = json.loads(file.read())

    with open('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/pid_crop_pad_bbox.json','r') as file:
        pid_crop_pad_bbox = json.loads(file.read())

    des_base_dir = '/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/openpose_img_resize'
    mkdir_if_absent(des_base_dir)

    src_kp_base_dir = '/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/openpose_json'
    kp_des_base_dir = '/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/openpose_json_resize'
    mkdir_if_absent(kp_des_base_dir)

    ratio = 3 / 4
    e_h, e_w = 1024, 768
    for idx, image_path in enumerate(
            glob.glob('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/openpose_img/*.jpg')):
        try:
            print(idx, image_path)
            name = image_path.strip().split('/')[-1]
            pid = int(name.replace('_rendered.jpg','.jpg').split('.')[0])
            if pid not in valid_pids:
                print('not valid', image_path)
                continue
            if os.path.exists(os.path.join(des_base_dir,name)):
                print('exist')
                continue
            crop_pad_bbox = pid_crop_pad_bbox.get(str(pid))
            image_pil = Image.open(image_path)
            w, h = image_pil.size

            img_array = np.asarray(image_pil)
            crop_bbox = crop_pad_bbox.get('crop_bbox')
            pad_bbox = crop_pad_bbox.get('pad_bbox')
            crop_image = img_array[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]]
            des_image = cv2.copyMakeBorder(crop_image, pad_bbox[1], pad_bbox[3], pad_bbox[0], pad_bbox[2],
                                           cv2.BORDER_REPLICATE)

            # print(crop_bbox, pad_bbox)
            des_h, des_w = des_image.shape[:2]
            output_img = Image.fromarray(np.asarray(des_image, dtype=np.uint8))
            output_img = output_img.resize((e_w,e_h),Image.Resampling.NEAREST)
            output_img.save(os.path.join(des_base_dir,name))
            # cv2.imwrite(os.path.join(des_base_dir, name.replace('.jpg','_origin.jpg')), image)
            with open(os.path.join(src_kp_base_dir,name.replace('_rendered.jpg','_keypoints.json')),'r') as file:
                json_result = json.loads(file.read())

            pose_keypoints_2d = json_result['people'][0]['pose_keypoints_2d']
            pose_keypoints_2d = np.asarray(pose_keypoints_2d).reshape(-1,3)
            des_pose_keypoints_2d = list()
            # kp_image = np.zeros((e_h,e_w,3))
            for kp in pose_keypoints_2d:
                x = round((kp[0] - crop_bbox[0] + pad_bbox[0]) * e_h / des_h)
                y = round((kp[1] - crop_bbox[1] + pad_bbox[1]) * e_w / des_w)
                des_pose_keypoints_2d.extend([x,y,kp[2]])
            # kp_image = cv2.circle(kp_image, (x, y), radius=10, color=(255, 255, 255), thickness=-1)
            json_result['people'][0]['pose_keypoints_2d'] = des_pose_keypoints_2d

            # cv2.imwrite(os.path.join(kp_des_base_dir,name),kp_image)
            with open(os.path.join(kp_des_base_dir,name.replace('_rendered.jpg','_keypoints.json')),'w') as file:
                file.write(json.dumps(json_result))

            # if idx > 100:
            #     break
        except:
            print(format_exc())

def resize_bottom_train_densepose_image():

    with open('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/fullbody_parse_2w_valid_pids.json','r') as file:
        valid_pids = json.loads(file.read())

    with open('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/pid_crop_pad_bbox.json','r') as file:
        pid_crop_pad_bbox = json.loads(file.read())

    des_base_dir = '/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/densepose_v1'
    mkdir_if_absent(des_base_dir)

    ratio = 3 / 4
    e_h, e_w = 1024, 768

    for idx, image_path in enumerate(
            glob.glob('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/image-densepose/*.jpg')):
        try:
            print(idx, image_path)
            name = image_path.strip().split('/')[-1]
            pid = int(name.split('.')[0])
            if pid not in valid_pids:
                print('not valid', image_path)
                continue
            if os.path.exists(os.path.join(des_base_dir,name)):
                print('exist')
                continue
            crop_pad_bbox = pid_crop_pad_bbox.get(str(pid))
            image_pil = Image.open(image_path)

            img_array = np.asarray(image_pil)
            crop_bbox = crop_pad_bbox.get('crop_bbox')
            pad_bbox = crop_pad_bbox.get('pad_bbox')
            crop_image = img_array[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]]
            des_image = cv2.copyMakeBorder(crop_image, pad_bbox[1], pad_bbox[3], pad_bbox[0], pad_bbox[2],
                                           cv2.BORDER_REPLICATE)

            output_img = Image.fromarray(np.asarray(des_image, dtype=np.uint8))
            output_img = output_img.resize((e_w,e_h),Image.Resampling.NEAREST)
            output_img.save(os.path.join(des_base_dir,name))

            # if idx > 100:
            #     break
        except:
            print(format_exc())

def process_atr_parse_face():
    data_path = '/home/vera/Dress_VITON/Dress_master_2501_test/dataset_0315/model/'

    os.makedirs(osp.join(data_path, 'parse_atr'), exist_ok=True)
    os.makedirs(osp.join(data_path, 'parse_lip'), exist_ok=True)
    for image_path in glob.glob(osp.join(data_path, 'parse_atr/*.png')):
        print(image_path)
        image_name = image_path.strip().split('/')[-1]
        atr_parse = Image.open(image_path)
        lip_parse = Image.open(osp.join(data_path, 'parse_lip', image_name))
        atr_parse_cv = np.asarray(atr_parse).copy()
        lip_parse_cv = np.asarray(lip_parse).copy()
        atr_face1_coor = np.where(atr_parse_cv==3)
        atr_face2_coor = np.where(atr_parse_cv == 11)
        lip_face1_coor = np.where(lip_parse_cv == 4)
        lip_face2_coor = np.where(lip_parse_cv == 13)
        neck_coor = set([i for i in zip(*atr_face1_coor)] + [i for i in zip(*atr_face2_coor)]) \
                    - set([i for i in zip(*lip_face1_coor)]+[i for i in zip(*lip_face2_coor)])
        neck_coor = tuple([np.asarray(i) for i in zip(*neck_coor)])
        bg_atr_parse_cv = np.asarray(atr_parse).copy()
        bg_atr_parse_cv[neck_coor] = 0
        output_img = Image.fromarray(bg_atr_parse_cv)
        palette = get_palette(18)
        output_img.putpalette(palette)
        output_img.save(os.path.join(data_path,'image-parse-v3',image_name))


if __name__ == '__main__':

    # get_valid_parse_and_bbox()
    # resize_bottom_train_full_image()
    # resize_bottom_train_parse_image()
    # resize_bottom_train_pose_image()
    # resize_bottom_train_densepose_image()
    process_atr_parse_face()
