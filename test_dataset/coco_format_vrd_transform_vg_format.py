import numpy as np
import torch
import h5py
import json
import os
from PIL import Image
from tqdm import tqdm
import copy
from shutil import copyfile

def transfer_saving(split='val', st_id=0):
    vrd_dir = './datasets/vrd/' + split + '_images/'
    new_dir = './datasets/vrd/images/'
    for f in tqdm(sorted(os.listdir(vrd_dir))):
        ext = f.split('.')[1]
        cnt = st_id + int(f.split('.')[0])
        if os.path.exists(new_dir + str(cnt) + '.jpg'):
            assert False
        
        if ext.find('png') >= 0 or ext.find('gif') >= 0:
            img = Image.open(vrd_dir + f).convert('RGB')
        else:
            copyfile(vrd_dir + f, new_dir + str(cnt) + '.jpg')

        if ext.find('gif') >= 0:
            img.save(new_dir + str(cnt) + '.jpg')
        elif ext.find('png') >= 0:
            img.save(new_dir + str(cnt) + '.jpg')



def box_transform(box):
    x = box[2]
    y = box[0]
    w = box[3] - box[2] + 1
    h = box[1] - box[0] + 1
    return [x, y, w, h]

def image_data_construct(rel_name_list, obj_name_list, split='val', st_id=0, \
 st_ent_id=0, st_rel_id=0):
    with open('./datasets/vrd/new_annotations_'+split+'.json', 'r') as f:
        rel_list = json.load(f)
        f.close()
    with open('./datasets/vrd/detections_'+split+'.json', 'r') as f:
        entdet_list = json.load(f)
        f.close()
    image_data_list = list()
    filename_id_map = dict()
    length_images = len(entdet_list['images'])
    new_entdet_list = dict()
    for i in entdet_list['images']:
        cur_dict = dict()
        cur_dict["image_id"] = st_id + int(i["file_name"].split('.')[0])
        cur_dict["url"] = None
        cur_dict["width"] = i["width"]
        cur_dict["height"] = i["height"]
        cur_dict["coco_id"] = None
        cur_dict["flickr_id"] = None
        image_data_list.append(cur_dict)
        new_ents_dict = dict()
        new_ents_dict['image_id'] = cur_dict["image_id"]
        new_ents_dict["objects"] = list()
        new_entdet_list[cur_dict["image_id"]] = new_ents_dict
        filename_id_map[i["id"]] = cur_dict["image_id"]
        length_images = max(length_images, int(i["file_name"].split('.')[0]))
    
    ent_rec_dict = dict()
    length_ent = 0
    for i in entdet_list['annotations']:
        cur_dict_pure_ent = dict()
        cur_dict_pure_ent['x'] = i['bbox'][0]
        cur_dict_pure_ent['y'] = i['bbox'][1]
        cur_dict_pure_ent['w'] = i['bbox'][2]
        cur_dict_pure_ent['h'] = i['bbox'][3]
        cur_dict_pure_ent['name'] = obj_name_list[i['category_id']]
        cur_dict_pure_ent['object_id'] = st_ent_id + i["id"]
        cur_dict_pure_ent['synsets'] = [cur_dict_pure_ent['name']+'.n.01']
        id_in_cur_dict = filename_id_map[i["image_id"]]
        new_entdet_list[id_in_cur_dict]["objects"].append(cur_dict_pure_ent)
        ent_rec_dict[(cur_dict_pure_ent['x'], cur_dict_pure_ent['y'], \
            cur_dict_pure_ent['w'], cur_dict_pure_ent['h'], \
            id_in_cur_dict)] = (cur_dict_pure_ent['object_id'], cur_dict_pure_ent['name'])
        length_ent = max(length_ent, i["id"])
    #print(len(ent_rec_dict))
    #for i, (k,v) in enumerate(ent_rec_dict.items()):
    #    #if i>50: break
    #    if k[-1] == 1441:
    #        print(k, v)
        
    new_entdet_list_ret = list()
    for k,v in new_entdet_list.items():
        new_entdet_list_ret.append(v)
    
    new_rel_list = list()
    relationship_id_cnt = st_rel_id
    for file_name, rels in rel_list.items():
        image_id = st_id + int(file_name.split('.')[0])
        cur_rel_dict = dict(image_id=image_id)
        cur_rel_dict['relationships'] = list()
        for rel_instance in rels:
            cur_cur_rel_dict = dict()
            predicate = rel_name_list[rel_instance['predicate']]
            rel_synsets = [predicate+'.n.01']
            obj_cls = obj_name_list[rel_instance['object']['category']]
            sbj_cls = obj_name_list[rel_instance['subject']['category']]
            obj_box = box_transform(rel_instance['object']['bbox'])
            sbj_box = box_transform(rel_instance['subject']['bbox'])
            obj_id, obj_cls = ent_rec_dict[(obj_box[0], obj_box[1], \
                obj_box[2], obj_box[3], image_id)]
            sbj_id, sbj_cls = ent_rec_dict[(sbj_box[0], sbj_box[1], \
                sbj_box[2], sbj_box[3], image_id)]
            obj_synsets = [obj_cls+'.n.01']
            sbj_synsets = [sbj_cls+'.n.01']
            cur_cur_rel_dict['relationship_id'] = relationship_id_cnt
            cur_cur_rel_dict['predicate'] = predicate
            cur_cur_rel_dict['synsets'] = rel_synsets
            cur_cur_rel_dict['subject'] = dict()
            cur_cur_rel_dict['subject']['object_id'] = sbj_id
            cur_cur_rel_dict['subject']['x'] = sbj_box[0]
            cur_cur_rel_dict['subject']['y'] = sbj_box[1]
            cur_cur_rel_dict['subject']['w'] = sbj_box[2]
            cur_cur_rel_dict['subject']['h'] = sbj_box[3]
            cur_cur_rel_dict['subject']['name'] = sbj_cls
            cur_cur_rel_dict['subject']['synsets'] = sbj_synsets
            cur_cur_rel_dict['object'] = dict()
            cur_cur_rel_dict['object']['object_id'] = obj_id
            cur_cur_rel_dict['object']['x'] = obj_box[0]
            cur_cur_rel_dict['object']['y'] = obj_box[1]
            cur_cur_rel_dict['object']['w'] = obj_box[2]
            cur_cur_rel_dict['object']['h'] = obj_box[3]
            cur_cur_rel_dict['object']['name'] = obj_cls
            cur_cur_rel_dict['object']['synsets'] = obj_synsets
            cur_rel_dict['relationships'].append(cur_cur_rel_dict)
            relationship_id_cnt += 1
        
        new_rel_list.append(cur_rel_dict)
    
    return length_images, length_ent, relationship_id_cnt, image_data_list, new_entdet_list_ret, new_rel_list
    
with open('./datasets/vrd/objects.json', 'r') as f:
    obj_name_list = json.load(f)
    f.close()
with open('./datasets/vrd/predicates.json', 'r') as f:
    rel_name_list = json.load(f)
    f.close()
length_images, length_ent, relationship_id_cnt, image_data_list_train, new_entdet_list_ret_train, new_rel_list_train = \
    image_data_construct(rel_name_list, obj_name_list, split='train', st_id=0, st_ent_id=0, st_rel_id=0)
_, _, _, image_data_list_val, new_entdet_list_ret_val, new_rel_list_val = \
    image_data_construct(rel_name_list, obj_name_list, split='val', st_id=length_images, \
        st_ent_id=length_ent, st_rel_id=relationship_id_cnt)
image_data_list = image_data_list_train + image_data_list_val
new_entdet_list_ret = new_entdet_list_ret_train + new_entdet_list_ret_val
new_rel_list = new_rel_list_train + new_rel_list_val
with open('/home/tengyao/scene-graph-TF-release/data_tools/VG/image_data.json', 'w') as f:
    json.dump(image_data_list, f)
    f.close()
with open('/home/tengyao/scene-graph-TF-release/data_tools/VG/objects.json', 'w') as f:
    json.dump(new_entdet_list_ret, f)
    f.close()
with open('/home/tengyao/scene-graph-TF-release/data_tools/VG/relationships.json', 'w') as f:
    json.dump(new_rel_list, f)
    f.close()
    
transfer_saving(split='train', st_id=0)
transfer_saving(split='val', st_id=length_images)