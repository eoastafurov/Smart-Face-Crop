import cv2
import numpy as np
import os


def hdr2jpg_clamp_tone_mapping(image_path, write_path):
    hdr = cv2.imread(image_path, flags=cv2.IMREAD_ANYDEPTH)
    # Simply clamp values to a 0-1 range for tone-mapping
    ldr = np.clip(hdr, 0, 1)
    # Color space conversion
    ldr = ldr**(1/2.2)
    # 0-255 remapping for bit-depth conversion
    ldr = 255.0 * ldr
    print(write_path)
    cv2.imwrite(write_path, ldr)

def map_hdr_to_jpg():

    imgs_path = './ColortypesDone1_HDR/'
    res_photo_path = './res_HDR'

    if not os.path.exists(res_photo_path):
        os.mkdir(res_photo_path)

    dirs = os.walk(imgs_path)
    for path_from_top, subdirs, files in dirs:
        for f in files:
            print("Working with " + f)
            if not f.endswith('.hdr'):
                continue
            filename = str(f).split('.')[0]
            image_path = str(path_from_top) + '/' + str(f)

            # print("Working with " + f)
            path = res_photo_path + path_from_top.split('.')[1] + '/'
            hdr2jpg_clamp_tone_mapping(image_path, path + '_after_hdr_' + filename + '.png')



map_hdr_to_jpg()