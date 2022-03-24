import numpy as np
import json

#annotations = json.load(open('datasets/openimages/open_image_v6/annotations/vrd-train-anno.json', 'r'))
annotations = json.load(open('datasets/openimages/open_image_v4/annotations/vrd-train-anno.json', 'r'))

num_img = len(annotations)

annotations = annotations[: num_img ]

empty_list = set()
for i, each in enumerate(annotations):
    if len(each['rel']) == 0:
        empty_list.add(i)
    if len(each['bbox']) == 0:
        empty_list.add(i)

print('empty relationship image num: ', len(empty_list))

entpairpro = np.zeros(58 - 1) #58-1, 602-1
relpro = np.zeros(9) #9, 30
max_class = 0
for i, anno in enumerate(annotations):

    if i in empty_list:
        continue

    boxes_i = np.array(anno['bbox'])
    gt_classes_i = np.array(anno['det_labels'], dtype=int)
    max_class = max(max_class, gt_classes_i.max())
    
    
    rels = np.array(anno['rel'], dtype=int)

    #gt_classes_i += 1
    #rels[:, -1] += 1
    
    tmp = rels[:, :-1].reshape(-1)
    
    rtmp = rels[:, -1]
    
    for j in tmp:
        entpairpro[gt_classes_i[j]] += 1.
    
    for j in rtmp:
        relpro[j] += 1.
    
entpairpro = entpairpro / (entpairpro.sum() + 1e-7)
entpairpro = entpairpro * 1000000
entpairpro = entpairpro.astype(np.long)
entpairpro = entpairpro.astype(np.float)
entpairpro = entpairpro / 1000000
entpairpro = entpairpro * (entpairpro >= 1e-8) + 1e-8 * (entpairpro < 1e-8)
entpairpro = entpairpro.tolist()
print(entpairpro)
print(max_class)

relpro = relpro / relpro.sum()
#relpro = np.log(relpro)
print('rel')
#for i in relpro: print(i)
print(relpro.tolist())

with open('datasets/openimages/open_image_v6/annotations/entpairpro.json', 'w') as f:
    json.dump(entpairpro, f)
    f.close()