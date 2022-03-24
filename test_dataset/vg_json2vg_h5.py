import argparse, json, string
from collections import Counter
import math

from math import floor
import h5py as h5
import numpy as np
import pprint






























def main(args):
    print('start')
    pprint.pprint(args)

    f = h5.File(args.h5_file, 'w')
    
    f.create_dataset('labels', data=encoded_label)
























if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imdb', default='VG/imdb_1024.h5', type=str)
    parser.add_argument('--object_input', default='VG/objects.json', type=str)
    parser.add_argument('--relationship_input', default='VG/relationships.json', type=str)
    parser.add_argument('--metadata_input', default='VG/image_data.json', type=str)
    parser.add_argument('--object_alias', default='VG/object_alias.txt', type=str)
    parser.add_argument('--pred_alias', default='VG/predicate_alias.txt', type=str)
    parser.add_argument('--object_list', default='VG/object_list.txt', type=str)
    parser.add_argument('--pred_list', default='VG/predicate_list.txt', type=str)
    parser.add_argument('--num_objects', default=100, type=int, help="set to 0 to disable filtering")
    parser.add_argument('--num_predicates', default=70, type=int, help="set to 0 to disable filtering")
    parser.add_argument('--min_box_area_frac', default=0.002, type=float)
    parser.add_argument('--json_file', default='VG-dicts.json')
    parser.add_argument('--h5_file', default='VG.h5')
    parser.add_argument('--load_frac', default=1, type=float)
    parser.add_argument('--use_input_split', default=False, type=bool)
    parser.add_argument('--train_frac', default=0.7, type=float)
    parser.add_argument('--val_frac', default=0.7, type=float)
    parser.add_argument('--shuffle', default=False, type=bool)

    args = parser.parse_args()
    main(args)