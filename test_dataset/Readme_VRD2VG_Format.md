## VRD_Reldn_Format 2 VG_Format

1. python ./test_dataset/coco_format_vrd_transform_vg_format.py
2. put generated images and jsons to *scene-graph-TF-release/data_tools/VG*
3. bash create_imdb.sh, then bash create_roidb.sh
4. put all generated files into datasets/vrd where the origin Reldn format vrd files are, and rename *VG*s into *vrd*s
5. ./datasets/vrd/0.generate_attribute_labels.ipynb to attach fake attributes