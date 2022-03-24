import numpy as np
import h5py
import json
import os
from PIL import Image
from tqdm import tqdm
import copy
from shutil import copyfile

def transfer_saving(main_path='/home/tengyao/reldnbench/data/'):
    oi_img_dir = main_path + 'openimages_v4/train/'
    new_dir = '/data1/tengyao/OpenImage_v4/selected_images/'
    file_name_list = list()
    with open(main_path + 'openimages_v4/rel/rel_only_annotations_val.json', 'r') as f:
        f_anno = json.load(f)
        f.close()
    for k, v in f_anno.items():
        file_name_list.append(k)
    with open(main_path + 'openimages_v4/rel/rel_only_annotations_train.json', 'r') as f:
        f_anno = json.load(f)
        f.close()
    for k, v in f_anno.items():
        file_name_list.append(k)
    
    for i in tqdm(file_name_list):
        if not os.path.exists(oi_img_dir + i):
            assert False
        copyfile(oi_img_dir + i, new_dir + i)
    
if __name__ == '__main__':
    transfer_saving()