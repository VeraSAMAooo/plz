import os.path as osp
import shutil
import math
import os
import glob
import json
import numpy as np
import cv2
from traceback import format_exc

def mkdir_if_absent(path):
    if not os.path.exists(path):
        os.makedirs(path)
    elif not os.path.isdir(path):
        # raise FileExistsError
        pass

def temp_filter_non_stand():
    kp_names_18 = ['nose','neck' ,'l_shoulder','l_elbow','l_wrist','r_shoulder','r_elbow','r_wrist',
                'l_hip','l_knee','l_ankle','r_hip','r_knee','r_ankle','l_eye', 'r_eye', 'l_ear', 'r_ear'
                ]
    kp_names_25 = ['nose','neck' ,'l_shoulder','l_elbow','l_wrist','r_shoulder','r_elbow','r_wrist','hip_middle'
                'l_hip','l_knee','l_ankle','r_hip','r_knee','r_ankle','l_eye', 'r_eye', 'l_ear', 'r_ear',
                   'r_feet','r_feet','r_feet','l_feet','l_feet','l_feet'
                ]
    #pose_dir = '/home/vera/myCode/db/dress_vton/resize_fullbody'
    src_pose_kp_dir = ('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/openpose_json')
    src_pose_img_dir = ('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/openpose_img')
    pass_fullbody_img = '/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/strand_fullbody'
    pose_fail_dir = '/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/pose_fail'
    pass_kp_dir = '/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/openpose_json_stand'
    pass_img_dir = '/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/openpose_img_stand'
    mkdir_if_absent(pass_kp_dir)
    mkdir_if_absent(pass_img_dir)
    mkdir_if_absent(pose_fail_dir)
    mkdir_if_absent(pass_fullbody_img)
    for idx,image_path in enumerate(glob.glob('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/image/*.jpg')):
        try:

            im_name = image_path.strip().split('/')[-1]
            pose_name = im_name.replace('.jpg', '_keypoints.json')
            print(image_path)

            if not osp.exists(osp.join(src_pose_kp_dir, pose_name)):
                shutil.copy(image_path, osp.join(pose_fail_dir, im_name))
                continue
            with open(osp.join(src_pose_kp_dir, pose_name), 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['people'][0]['pose_keypoints_2d']
                pose_data = np.array(pose_data)
                keypoints = pose_data.reshape((-1, 3))[:, :2]

            l_shoulder, r_shoulder = keypoints[2], keypoints[5]
            l_hip, r_hip = keypoints[9], keypoints[12]
            l_knee, r_knee = keypoints[10], keypoints[13]
            l_ankle, r_ankle = keypoints[11], keypoints[14]
            l_knee_hip_x,l_knee_hip_y = l_knee[0] - l_hip[0],l_knee[1] - l_hip[1]
            r_knee_hip_x, r_knee_hip_y = r_knee[0] - r_hip[0], r_knee[1] - r_hip[1]
            hip_width = abs(r_hip[0] - l_hip[0])
            shoulder_hip_len = max((l_hip[1]-l_shoulder[1]), (r_hip[1]-r_shoulder[1]))
            l_knee_hip_len,r_knee_hip_len = (l_knee[1]-l_hip[1]),(r_knee[1] - r_hip[1])
            l_ankle_knee_len, r_ankle_knee_len = (l_ankle[1] - l_knee[1]), (r_ankle[1] - r_knee[1])
            # print(shoulder_hip_len,l_knee_hip_len/shoulder_hip_len,r_knee_hip_len/shoulder_hip_len,l_ankle_knee_len/shoulder_hip_len,r_ankle_knee_len/shoulder_hip_len,hip_width/shoulder_hip_len,
            #       l_knee_hip_x,(l_knee_hip_y/l_knee_hip_x),r_knee_hip_x,(r_knee_hip_y/r_knee_hip_x))
            if (l_knee_hip_len < 0.5 * shoulder_hip_len) or (r_knee_hip_len < 0.5 * shoulder_hip_len) or \
                    (l_ankle_knee_len < 0.5 * shoulder_hip_len) or (r_ankle_knee_len < 0.5 * shoulder_hip_len) or \
                    hip_width < 0.25*shoulder_hip_len or \
                    ((l_knee_hip_len > 1.2 * shoulder_hip_len) and r_knee_hip_len > 1.2 * shoulder_hip_len) or \
                    ((l_ankle_knee_len > 1.2 * shoulder_hip_len) and r_ankle_knee_len > 1.2 * shoulder_hip_len):
                # print('fail')
                shutil.copy(image_path,osp.join(pose_fail_dir,im_name))
            else:
                if (l_knee_hip_x < 0 and abs(l_knee_hip_y/l_knee_hip_x) < math.tan(45*math.pi/180) and r_knee_hip_x < 0 and abs(r_knee_hip_y/r_knee_hip_x) < math.tan(60*math.pi/180)) or \
                        (l_knee_hip_x > 0 and abs(l_knee_hip_y / l_knee_hip_x) < math.tan(
                            60 * math.pi / 180) and r_knee_hip_x > 0 and abs(
                            r_knee_hip_y/r_knee_hip_x) < math.tan(45 * math.pi / 180)):
                    # print('fail angle')
                    shutil.copy(image_path, osp.join(pose_fail_dir, im_name))
                else:
                    image = cv2.imread(image_path)
                    h,w = image.shape[:2]
                    if (h - l_ankle[1]) < max(10,h/30) or (h - r_ankle[1]) < max(10,h/30):
                        # print('fail',h,(h - l_ankle[1])/h,(h - r_ankle[1])/h)
                        shutil.copy(image_path, osp.join(pose_fail_dir, im_name))
                    else:
                        # print('pass',h,(h - l_ankle[1])/h,(h - r_ankle[1])/h)
                        #shutil.copy(osp.join(src_pose_kp_dir, pose_name), osp.join(pass_kp_dir, pose_name))
                        rend_img_name = im_name.replace('.jpg','_rendered.jpg')
                        #shutil.copy(osp.join(src_pose_img_dir, rend_img_name), osp.join(pass_img_dir, rend_img_name))
                        shutil.copy(image_path, osp.join(pass_fullbody_img, im_name))

        except:
            print(format_exc())

if __name__ == '__main__':
    temp_filter_non_stand()