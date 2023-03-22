from PIL import Image
import numpy as np
import os
# path = '/home/vera/bttest/old'
#
# dstpath = '/home/vera/bttest/image-parse-agnostic-v3.2' # Destination Folder
#
# if not os.path.exists(dstpath):   # create path if the path is not exit
#     os.makedirs(dstpath)
#
# files = os.listdir(path)
# for img in files:
#     img1 = img.convert("P")
#     img1.save(os.path.join(dstpath, img))

image1 = Image.open('/home/vera/Dress_VITON/Dress_generator_test/visualization/test_outputs/tocg3/2335067.jpg_8215110.jpg')
image2 = Image.open('/home/vera/Dress_VITON/Dress_Condition_test/visualization/test_outputs/tocg3/141220.jpg_7036584.jpg')
# new_im1 = image1.resize([768, 1024])
image_array1 = np.asarray(image1)
image_array2 = np.asarray(image2)
# new_im1.save('/home/vera/Dress_VITON/Dress_Condition_test/data/fake_warped_cloth_new/141703.jpg')
print('done')