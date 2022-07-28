'''
    parse coco file to crop tooth images
    vectormaster
'''

import json
import os
from os import path
from functools import reduce
import cv2

COCO_FILE = 'arch_upper_16cls_test.json'
INPUT_DIR = 'test_upper_gray'
OUTPUT_DIR = 'output'
N = 3
TOOTH_ID_SEQ = [43, 14, 11, 9, 7, 4, 12, 13, 2, 6, 1, 8, 0, 16, 17, 45]     # tooth number order 18,17,...,11,21,22,...,28

def convert_to_min_max_box(bbox_coco):
    return {
        'xmin': bbox_coco[0],
        'ymin': bbox_coco[1],
        'xmax': bbox_coco[0] + bbox_coco[2],
        'ymax': bbox_coco[1] + bbox_coco[3]
    }

def get_big_box(n_boxes):
    return {
        'xmin': int(min(map(lambda box: box['xmin'], n_boxes))),
        'ymin': int(min(map(lambda box: box['ymin'], n_boxes))),
        'xmax': int(max(map(lambda box: box['xmax'], n_boxes))),
        'ymax': int(max(map(lambda box: box['ymax'], n_boxes))),
    }


def crop_images(coco_file=COCO_FILE, input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, N=3):
    f = open(COCO_FILE)
    raw_data = json.load(f)
    f.close()

    image_file_dict = {}            # image_id => file_name
    category_id_to_name_dict = {}   # category_id => tooth_name
    bboxes = {}                     # image_id => category_id => bbox

    for image in raw_data['images']:
        image_file_dict[image['id']] = image['file_name']

    for cat in raw_data['categories']:
        category_id_to_name_dict[cat['id']] = cat['name'][6:]

    # for tooth_id in TOOTH_ID_SEQ:
    #     print(category_id_to_name_dict[tooth_id])

    for annotation in raw_data['annotations']:
        box = convert_to_min_max_box(annotation['bbox'])
        try:
            bboxes[annotation['image_id']][annotation['category_id']] = box
        except:
            bboxes[annotation['image_id']] = {annotation['category_id']: box}

    try:
        os.mkdir(output_dir) 
    except:
        pass

    idx = 0
    for (image_id, image) in bboxes.items():
        image_file_name = image_file_dict[image_id]
        image_mat = cv2.imread(path.join(input_dir, image_file_name))
        for i in range(len(TOOTH_ID_SEQ) - 2):
            n = 0
            n_near_teeth_boxes = []
            n_near_teeth_ids = []
            for j in range(i, len(TOOTH_ID_SEQ)):
                if n == N:
                    break
                try:
                    n_near_teeth_boxes.append(image[TOOTH_ID_SEQ[j]])
                    n_near_teeth_ids.append(category_id_to_name_dict[TOOTH_ID_SEQ[j]])
                    n += 1
                except:
                    pass
            if n < N:
                continue
            big_box = get_big_box(n_near_teeth_boxes)
            cv2.imwrite(path.join(output_dir ,image_file_name[:-4] + "_" + str(idx) + "_" + reduce(lambda a, b: str(a) + "-" + str(b), n_near_teeth_ids) + ".jpg"), 
                image_mat[big_box['ymin']:big_box['ymax'],big_box['xmin']:big_box['xmax'],:])
            idx += 1

if __name__ == "__main__":
    crop_images(coco_file=COCO_FILE, input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, N=3)