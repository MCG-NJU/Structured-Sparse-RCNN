import numpy as np
import torch
import h5py

#with h5py.File('./datasets/vg/VG-SGG-with-attri.h5', 'r') as f:
with h5py.File('./datasets/vrd/vrd-SGG.h5', 'r') as f:
    print(len(f.keys()))
    for key in f.keys():
        print(key)
        print(f[key].name)
        print(f[key])
        if key == 'split':
            k0,k1,k2=0,0,0
            for i,k in enumerate(f[key]):
                if k == 0: k0+=1
                elif k== 1: k1+=1
                elif k==2: k2+=1
            print()
            print(k0,k1,k2)
    f.close()

print()
print()
print()
assert False
    
with h5py.File('./datasets/vg/old_h5/VG-SGG.h5', 'r') as f:
    for key in f.keys():
        print(f[key])
        print(key)
        print(f[key].name)
    f.close()