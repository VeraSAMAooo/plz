import os
from PIL import Image

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def mkdir_if_absent(path):
    if not os.path.exists(path):
        os.makedirs(path)
    elif not os.path.isdir(path):
        # raise FileExistsError
        pass


img_dir = os.listdir('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/image')
dress_dir = os.listdir('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/dress/cloth')

list3 = intersection(img_dir,dress_dir)

# with open('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/intersection_list1.txt', 'w') as fp:
#     for item in list3:
#         fp.write("%s\n" % item)

#if in list -> save

imagelist = os.listdir('/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/image')
image_dir = '/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/image'

##des
des_img = '/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/image_v1'
des_gray_path = '/home/vera/Dress_VITON/Dress_master_2501_test/dataset_roundneck/model/gray'
# transparency_path = 'demo_transparency'
mkdir_if_absent(des_img)
mkdir_if_absent(des_gray_path)

# print(imagelist[0])
for i in range(len(imagelist)):
    if imagelist[i] in list3:
