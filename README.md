# Structured Sparse R-CNN for Direct Scene Graph Generation

Our paper [Structured Sparse R-CNN for Direct Scene Graph Generation](https://arxiv.org/abs/2106.10815) has been accepted by CVPR 2022.


## Requirements

Environments: python 3.8, cuda 10.1, pytorch 1.7.1

To install requirements:

```bash
conda create --name scene_graph_benchmark
conda activate scene_graph_benchmark

pip install --user ipython
pip install --user scipy
pip install --user h5py
pip install --user pyyaml
pip install --user yacs
pip install --user scipy
pip install --user h5py
pip install --user tqdm
pip install --user opencv-python

pip install --user ninja yacs cython matplotlib tqdm opencv-python overrides


conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch

git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# put our code SGGbench into your directory, such as /home/username
cd /home/username/SGGbench 
python setup.py build develop

```

## Datasets

### VG

For the dataset and pretrained backbone weights preparation, please follow: [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)

### OI v4, v6

For the datasets and pretrained backbone weights preparation, please follow: [BGNN-SGG](https://github.com/SHTUPLUS/PySGG). Actually, [BGNN-SGG](https://github.com/SHTUPLUS/PySGG) is also compatible with [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).


## Training

### VG

To train the model with 300 triplet queries in the paper, run this command: (8 RTX 2080ti 11G)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 10020 --nproc_per_node=8 tools/relation_train_net.py --config-file "configs/simrel_e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 8 SOLVER.VAL_PERIOD 12000 SOLVER.MAX_ITER 80000 SOLVER.STEPS '(47000, 64000)' SOLVER.BASE_LR 0.000008 SOLVER.OPTIMIZER "ADAMW" MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR Outputs/real025kl_newe2rposition_fullobjbranch_300q_smalldyconv_prequeryaug9_norel_hardlabel MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM 1024 MODEL.SimrelRCNN.ENABLE_BG_OBJ False MODEL.SimrelRCNN.ENABLE_REL_X2Y True MODEL.SimrelRCNN.REL_DIM 256 MODEL.SimrelRCNN.NUM_PROPOSALS 300 MODEL.SimrelRCNN.NUM_HEADS 6 MODEL.SimrelRCNN.REL_STACK_NUM 6 MODEL.SimrelRCNN.TRIPLET_MASK_WEIGHT 1.0 MODEL.SimrelRCNN.FREEZE_BACKBONE True MODEL.SimrelRCNN.CROSS_OBJ_FEAT_FUSION False MODEL.SimrelRCNN.CLASS_WEIGHT 1.333 MODEL.SimrelRCNN.L1_WEIGHT 5.0 MODEL.SimrelRCNN.GIOU_WEIGHT 2.0 GLOVE_DIR '' MODEL.SimrelRCNN.ENABLE_FREQ False MODEL.SimrelRCNN.ENABLE_QUERY_REVERSE False MODEL.SimrelRCNN.USE_REFINE_OBJ_FEATURE True MODEL.SimrelRCNN.FREEZE_PUREE_OBJDET False MODEL.SimrelRCNN.PURE_ENT_NUM_PROPOSALS 100 MODEL.SimrelRCNN.ENABLE_MASK_BRANCH False MODEL.SimrelRCNN.KL_BRANCH_WEIGHT 0.25 MODEL.SimrelRCNN.ENABLE_KL_BRANCH True MODEL.SimrelRCNN.POSI_ENCODE_DIM 64 MODEL.SimrelRCNN.REL_CLASS_WEIGHT 1.334 MODEL.SimrelRCNN.PURE_ENT_CLASS_WEIGHT 2.0 MODEL.SimrelRCNN.PURE_ENT_GIOU_WEIGHT 2.0 MODEL.SimrelRCNN.PURE_ENT_L1_WEIGHT 5.0 MODEL.SimrelRCNN.AUXILIARY_BRANCH True MODEL.SimrelRCNN.AUXILIARY_BRANCH_SELECT_ENT_MAX_NUM 25 MODEL.SimrelRCNN.AUXILIARY_BRANCH_START 11 MODEL.SimrelRCNN.ENABLE_ENT_PROP True MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS False MODEL.SimrelRCNN.USE_CROSS_RANK False MODEL.SimrelRCNN.DISABLE_KQ_FUSION_SELFATTEN False MODEL.SimrelRCNN.DIM_ENT_PRE_CLS 256 MODEL.SimrelRCNN.DIM_ENT_PRE_REG 256 MODEL.SimrelRCNN.ONE_REL_CONV True MODEL.SimrelRCNN.ENABLE_BATCH_REDUCTION True MODEL.SimrelRCNN.USE_HARD_LABEL_KLMATCH True MODEL.SimrelRCNN.DISABLE_OBJ2REL_LOSS True
```

To train the model with 800 triplet queries in the paper, run this command: (8 V100 32G)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 10020 --nproc_per_node=8 tools/relation_train_net.py --config-file "configs/simrel_e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 8 SOLVER.VAL_PERIOD 12000 SOLVER.MAX_ITER 80000 SOLVER.STEPS '(47000, 64000)' SOLVER.BASE_LR 0.000008 SOLVER.OPTIMIZER "ADAMW" MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR Outputs/real025kl_newe2rposition_fullobjbranch_800q_smalldyconv_prequeryaug9_norel_hardlabel MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM 1024 MODEL.SimrelRCNN.ENABLE_BG_OBJ False MODEL.SimrelRCNN.ENABLE_REL_X2Y True MODEL.SimrelRCNN.REL_DIM 256 MODEL.SimrelRCNN.NUM_PROPOSALS 800 MODEL.SimrelRCNN.NUM_HEADS 6 MODEL.SimrelRCNN.REL_STACK_NUM 6 MODEL.SimrelRCNN.TRIPLET_MASK_WEIGHT 1.0 MODEL.SimrelRCNN.FREEZE_BACKBONE True MODEL.SimrelRCNN.CROSS_OBJ_FEAT_FUSION False MODEL.SimrelRCNN.CLASS_WEIGHT 1.333 MODEL.SimrelRCNN.L1_WEIGHT 5.0 MODEL.SimrelRCNN.GIOU_WEIGHT 2.0 GLOVE_DIR '' MODEL.SimrelRCNN.ENABLE_FREQ False MODEL.SimrelRCNN.ENABLE_QUERY_REVERSE False MODEL.SimrelRCNN.USE_REFINE_OBJ_FEATURE True MODEL.SimrelRCNN.FREEZE_PUREE_OBJDET False MODEL.SimrelRCNN.PURE_ENT_NUM_PROPOSALS 100 MODEL.SimrelRCNN.ENABLE_MASK_BRANCH False MODEL.SimrelRCNN.KL_BRANCH_WEIGHT 0.25 MODEL.SimrelRCNN.ENABLE_KL_BRANCH True MODEL.SimrelRCNN.POSI_ENCODE_DIM 64 MODEL.SimrelRCNN.REL_CLASS_WEIGHT 1.334 MODEL.SimrelRCNN.PURE_ENT_CLASS_WEIGHT 2.0 MODEL.SimrelRCNN.PURE_ENT_GIOU_WEIGHT 2.0 MODEL.SimrelRCNN.PURE_ENT_L1_WEIGHT 5.0 MODEL.SimrelRCNN.AUXILIARY_BRANCH True MODEL.SimrelRCNN.AUXILIARY_BRANCH_SELECT_ENT_MAX_NUM 25 MODEL.SimrelRCNN.AUXILIARY_BRANCH_START 11 MODEL.SimrelRCNN.ENABLE_ENT_PROP True MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS False MODEL.SimrelRCNN.USE_CROSS_RANK False MODEL.SimrelRCNN.DISABLE_KQ_FUSION_SELFATTEN False MODEL.SimrelRCNN.DIM_ENT_PRE_CLS 256 MODEL.SimrelRCNN.DIM_ENT_PRE_REG 256 MODEL.SimrelRCNN.ONE_REL_CONV True MODEL.SimrelRCNN.ENABLE_BATCH_REDUCTION True MODEL.SimrelRCNN.USE_HARD_LABEL_KLMATCH True MODEL.SimrelRCNN.DISABLE_OBJ2REL_LOSS True
```

### OI V4

To train the model with 300 triplet queries in the paper, run this command: (8 RTX 2080ti 11G)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 10020 --nproc_per_node=8 tools/relation_train_net.py --config-file "configs/oiv4_simrel_e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 8 SOLVER.VAL_PERIOD 12000 SOLVER.MAX_ITER 80000 SOLVER.STEPS '(47000, 64000)' SOLVER.BASE_LR 0.000008 SOLVER.OPTIMIZER "ADAMW" MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/iov4_pretrain_faster_rcnn/model_final.pth OUTPUT_DIR Outputs/oiv4_real025kl_newe2rposition_fullobjbranch_300q_smalldyconv_prequeryaug9_norel_hardlabel MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM 1024 MODEL.SimrelRCNN.ENABLE_BG_OBJ False MODEL.SimrelRCNN.ENABLE_REL_X2Y True MODEL.SimrelRCNN.REL_DIM 256 MODEL.SimrelRCNN.NUM_PROPOSALS 300 MODEL.SimrelRCNN.NUM_HEADS 6 MODEL.SimrelRCNN.REL_STACK_NUM 6 MODEL.SimrelRCNN.TRIPLET_MASK_WEIGHT 1.0 MODEL.SimrelRCNN.FREEZE_BACKBONE True MODEL.SimrelRCNN.CROSS_OBJ_FEAT_FUSION False MODEL.SimrelRCNN.CLASS_WEIGHT 1.333 MODEL.SimrelRCNN.L1_WEIGHT 5.0 MODEL.SimrelRCNN.GIOU_WEIGHT 2.0 GLOVE_DIR '' MODEL.SimrelRCNN.ENABLE_FREQ False MODEL.SimrelRCNN.ENABLE_QUERY_REVERSE False MODEL.SimrelRCNN.USE_REFINE_OBJ_FEATURE True MODEL.SimrelRCNN.FREEZE_PUREE_OBJDET False MODEL.SimrelRCNN.PURE_ENT_NUM_PROPOSALS 100 MODEL.SimrelRCNN.ENABLE_MASK_BRANCH False MODEL.SimrelRCNN.KL_BRANCH_WEIGHT 0.25 MODEL.SimrelRCNN.ENABLE_KL_BRANCH True MODEL.SimrelRCNN.POSI_ENCODE_DIM 64 MODEL.SimrelRCNN.REL_CLASS_WEIGHT 1.334 MODEL.SimrelRCNN.PURE_ENT_CLASS_WEIGHT 2.0 MODEL.SimrelRCNN.PURE_ENT_GIOU_WEIGHT 2.0 MODEL.SimrelRCNN.PURE_ENT_L1_WEIGHT 5.0 MODEL.SimrelRCNN.AUXILIARY_BRANCH True MODEL.SimrelRCNN.AUXILIARY_BRANCH_SELECT_ENT_MAX_NUM 25 MODEL.SimrelRCNN.AUXILIARY_BRANCH_START 11 MODEL.SimrelRCNN.ENABLE_ENT_PROP True MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS False MODEL.SimrelRCNN.USE_CROSS_RANK False MODEL.SimrelRCNN.DISABLE_KQ_FUSION_SELFATTEN False MODEL.SimrelRCNN.DIM_ENT_PRE_CLS 1024 MODEL.SimrelRCNN.DIM_ENT_PRE_REG 256 MODEL.SimrelRCNN.ONE_REL_CONV True MODEL.SimrelRCNN.ENABLE_BATCH_REDUCTION True MODEL.SimrelRCNN.USE_HARD_LABEL_KLMATCH True MODEL.SimrelRCNN.DISABLE_OBJ2REL_LOSS True
```

### OI V6

To train the model with 300 triplet queries in the paper, run this command: (8 RTX 2080ti 11G)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 10020 --nproc_per_node=8 tools/relation_train_net.py --config-file "configs/oiv6_simrel_e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 8 SOLVER.VAL_PERIOD 12000 SOLVER.MAX_ITER 80000 SOLVER.STEPS '(47000, 64000)' SOLVER.BASE_LR 0.000008 SOLVER.OPTIMIZER "ADAMW" MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/iov6_pretrain_faster_rcnn/model_final.pth OUTPUT_DIR Outputs/oiv6_real025kl_newe2rposition_fullobjbranch_300q_smalldyconv_prequeryaug9_norel_hardlabel MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM 1024 MODEL.SimrelRCNN.ENABLE_BG_OBJ False MODEL.SimrelRCNN.ENABLE_REL_X2Y True MODEL.SimrelRCNN.REL_DIM 256 MODEL.SimrelRCNN.NUM_PROPOSALS 300 MODEL.SimrelRCNN.NUM_HEADS 6 MODEL.SimrelRCNN.REL_STACK_NUM 6 MODEL.SimrelRCNN.TRIPLET_MASK_WEIGHT 1.0 MODEL.SimrelRCNN.FREEZE_BACKBONE True MODEL.SimrelRCNN.CROSS_OBJ_FEAT_FUSION False MODEL.SimrelRCNN.CLASS_WEIGHT 1.333 MODEL.SimrelRCNN.L1_WEIGHT 5.0 MODEL.SimrelRCNN.GIOU_WEIGHT 2.0 GLOVE_DIR '' MODEL.SimrelRCNN.ENABLE_FREQ False MODEL.SimrelRCNN.ENABLE_QUERY_REVERSE False MODEL.SimrelRCNN.USE_REFINE_OBJ_FEATURE True MODEL.SimrelRCNN.FREEZE_PUREE_OBJDET False MODEL.SimrelRCNN.PURE_ENT_NUM_PROPOSALS 100 MODEL.SimrelRCNN.ENABLE_MASK_BRANCH False MODEL.SimrelRCNN.KL_BRANCH_WEIGHT 0.25 MODEL.SimrelRCNN.ENABLE_KL_BRANCH True MODEL.SimrelRCNN.POSI_ENCODE_DIM 64 MODEL.SimrelRCNN.REL_CLASS_WEIGHT 1.334 MODEL.SimrelRCNN.PURE_ENT_CLASS_WEIGHT 2.0 MODEL.SimrelRCNN.PURE_ENT_GIOU_WEIGHT 2.0 MODEL.SimrelRCNN.PURE_ENT_L1_WEIGHT 5.0 MODEL.SimrelRCNN.AUXILIARY_BRANCH True MODEL.SimrelRCNN.AUXILIARY_BRANCH_SELECT_ENT_MAX_NUM 25 MODEL.SimrelRCNN.AUXILIARY_BRANCH_START 11 MODEL.SimrelRCNN.ENABLE_ENT_PROP True MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS False MODEL.SimrelRCNN.USE_CROSS_RANK False MODEL.SimrelRCNN.DISABLE_KQ_FUSION_SELFATTEN False MODEL.SimrelRCNN.DIM_ENT_PRE_CLS 1024 MODEL.SimrelRCNN.DIM_ENT_PRE_REG 256 MODEL.SimrelRCNN.ONE_REL_CONV True MODEL.SimrelRCNN.ENABLE_BATCH_REDUCTION True MODEL.SimrelRCNN.USE_HARD_LABEL_KLMATCH True MODEL.SimrelRCNN.DISABLE_OBJ2REL_LOSS True
```

## Evaluation

### VG

#### 300 queries

To evaluate my model, run:

```eval

```

>📋  Before testing, create a new directory, named 'test_1024_8bs_300q_100_more_abla_10', in ./SGGbench/Outputs/ .

To evaluate my model with TDE, run:

```eval

```

>📋  Before testing, create a new directory, named 'causal_1024_8bs_300q_100_more_abla_10', in ./SGGbench/Outputs/ .

To evaluate my model with our LA, run:

```eval

```

>📋  Before testing, create a new directory, named 'LA03_real025kl_newe2rposition_fullobjbranch_800q_smalldyconv_prequeryaug9_norel_hardlabel', in ./SGGbench/Outputs/ .

#### 800 queries

To evaluate my model, run:

```eval

```

>📋  Before testing, create a new directory, named 'test_1024_8bs_300q_100_more_abla_10', in ./SGGbench/Outputs/ .

To evaluate my model with TDE, run:

```eval

```

>📋  Before testing, create a new directory, named 'causal_1024_8bs_300q_100_more_abla_10', in ./SGGbench/Outputs/ .

To evaluate my model with our LA, run:

```eval
CUDA_VISIBLE_DEVICES=3,5,6,7 python -m torch.distributed.launch --master_port 10029 --nproc_per_node=4 tools/relation_test_net.py --config-file "configs/simrel_e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 4 SOLVER.VAL_PERIOD 12000 SOLVER.MAX_ITER 80000 SOLVER.STEPS '(47000, 64000)' SOLVER.BASE_LR 0.000008 SOLVER.OPTIMIZER "ADAMW" MODEL.WEIGHT Outputs/real025kl_newe2rposition_fullobjbranch_800q_smalldyconv_prequeryaug9_norel_hardlabel/model_final.pth OUTPUT_DIR Outputs/LA03_real025kl_newe2rposition_fullobjbranch_800q_smalldyconv_prequeryaug9_norel_hardlabel MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM 1024 MODEL.SimrelRCNN.ENABLE_BG_OBJ False MODEL.SimrelRCNN.ENABLE_REL_X2Y True MODEL.SimrelRCNN.REL_DIM 256 MODEL.SimrelRCNN.NUM_PROPOSALS 800 MODEL.SimrelRCNN.NUM_HEADS 6 MODEL.SimrelRCNN.REL_STACK_NUM 6 MODEL.SimrelRCNN.TRIPLET_MASK_WEIGHT 1.0 MODEL.SimrelRCNN.FREEZE_BACKBONE True MODEL.SimrelRCNN.CROSS_OBJ_FEAT_FUSION False MODEL.SimrelRCNN.CLASS_WEIGHT 1.333 MODEL.SimrelRCNN.L1_WEIGHT 5.0 MODEL.SimrelRCNN.GIOU_WEIGHT 2.0 GLOVE_DIR '' MODEL.SimrelRCNN.ENABLE_FREQ False MODEL.SimrelRCNN.ENABLE_QUERY_REVERSE False MODEL.SimrelRCNN.USE_REFINE_OBJ_FEATURE True MODEL.SimrelRCNN.FREEZE_PUREE_OBJDET False MODEL.SimrelRCNN.PURE_ENT_NUM_PROPOSALS 100 MODEL.SimrelRCNN.ENABLE_MASK_BRANCH False MODEL.SimrelRCNN.KL_BRANCH_WEIGHT 0.25 MODEL.SimrelRCNN.ENABLE_KL_BRANCH True MODEL.SimrelRCNN.POSI_ENCODE_DIM 64 MODEL.SimrelRCNN.REL_CLASS_WEIGHT 1.334 MODEL.SimrelRCNN.PURE_ENT_CLASS_WEIGHT 2.0 MODEL.SimrelRCNN.PURE_ENT_GIOU_WEIGHT 2.0 MODEL.SimrelRCNN.PURE_ENT_L1_WEIGHT 5.0 MODEL.SimrelRCNN.AUXILIARY_BRANCH True MODEL.SimrelRCNN.AUXILIARY_BRANCH_SELECT_ENT_MAX_NUM 25 MODEL.SimrelRCNN.AUXILIARY_BRANCH_START 11 MODEL.SimrelRCNN.ENABLE_ENT_PROP True MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS False MODEL.SimrelRCNN.USE_CROSS_RANK False MODEL.SimrelRCNN.DISABLE_KQ_FUSION_SELFATTEN False MODEL.SimrelRCNN.DIM_ENT_PRE_CLS 256 MODEL.SimrelRCNN.DIM_ENT_PRE_REG 256 MODEL.SimrelRCNN.ONE_REL_CONV True MODEL.SimrelRCNN.ENABLE_BATCH_REDUCTION True MODEL.SimrelRCNN.USE_HARD_LABEL_KLMATCH True MODEL.SimrelRCNN.DISABLE_OBJ2REL_LOSS True MODEL.SimrelRCNN.REL_LOGITS_ADJUSTMENT True MODEL.SimrelRCNN.LOGIT_ADJ_TAU 0.3
```

>📋  Before testing, create a new directory, named 'LA03_real025kl_newe2rposition_fullobjbranch_800q_smalldyconv_prequeryaug9_norel_hardlabel', in ./SGGbench/Outputs/ .

### OI V4

#### 300 queries

To evaluate my model, run:

```eval

```

>📋  Before testing, create a new directory, named 'test_1024_8bs_300q_100_more_abla_10', in ./SGGbench/Outputs/ .

To evaluate my model with our LA, run:

```eval

```

>📋  Before testing, create a new directory, named 'LA03_real025kl_newe2rposition_fullobjbranch_800q_smalldyconv_prequeryaug9_norel_hardlabel', in ./SGGbench/Outputs/ .

### OI V6

#### 300 queries

To evaluate my model, run:

```eval
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 10020 --nproc_per_node=8 tools/relation_test_net.py --config-file "configs/oiv6_simrel_e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 8 SOLVER.VAL_PERIOD 12000 SOLVER.MAX_ITER 80000 SOLVER.STEPS '(47000, 64000)' SOLVER.BASE_LR 0.000008 SOLVER.OPTIMIZER "ADAMW" MODEL.WEIGHT Outputs/oiv6_real025kl_newe2rposition_fullobjbranch_300q_smalldyconv_prequeryaug9_norel_hardlabel/model_final.pth OUTPUT_DIR Outputs/eva_oiv6_real025kl_newe2rposition_fullobjbranch_300q_smalldyconv_prequeryaug9_norel_hardlabel MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM 1024 MODEL.SimrelRCNN.ENABLE_BG_OBJ False MODEL.SimrelRCNN.ENABLE_REL_X2Y True MODEL.SimrelRCNN.REL_DIM 256 MODEL.SimrelRCNN.NUM_PROPOSALS 300 MODEL.SimrelRCNN.NUM_HEADS 6 MODEL.SimrelRCNN.REL_STACK_NUM 6 MODEL.SimrelRCNN.TRIPLET_MASK_WEIGHT 1.0 MODEL.SimrelRCNN.FREEZE_BACKBONE True MODEL.SimrelRCNN.CROSS_OBJ_FEAT_FUSION False MODEL.SimrelRCNN.CLASS_WEIGHT 1.333 MODEL.SimrelRCNN.L1_WEIGHT 5.0 MODEL.SimrelRCNN.GIOU_WEIGHT 2.0 GLOVE_DIR '' MODEL.SimrelRCNN.ENABLE_FREQ False MODEL.SimrelRCNN.ENABLE_QUERY_REVERSE False MODEL.SimrelRCNN.USE_REFINE_OBJ_FEATURE True MODEL.SimrelRCNN.FREEZE_PUREE_OBJDET False MODEL.SimrelRCNN.PURE_ENT_NUM_PROPOSALS 100 MODEL.SimrelRCNN.ENABLE_MASK_BRANCH False MODEL.SimrelRCNN.KL_BRANCH_WEIGHT 0.25 MODEL.SimrelRCNN.ENABLE_KL_BRANCH True MODEL.SimrelRCNN.POSI_ENCODE_DIM 64 MODEL.SimrelRCNN.REL_CLASS_WEIGHT 1.334 MODEL.SimrelRCNN.PURE_ENT_CLASS_WEIGHT 2.0 MODEL.SimrelRCNN.PURE_ENT_GIOU_WEIGHT 2.0 MODEL.SimrelRCNN.PURE_ENT_L1_WEIGHT 5.0 MODEL.SimrelRCNN.AUXILIARY_BRANCH True MODEL.SimrelRCNN.AUXILIARY_BRANCH_SELECT_ENT_MAX_NUM 25 MODEL.SimrelRCNN.AUXILIARY_BRANCH_START 11 MODEL.SimrelRCNN.ENABLE_ENT_PROP True MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS False MODEL.SimrelRCNN.USE_CROSS_RANK False MODEL.SimrelRCNN.DISABLE_KQ_FUSION_SELFATTEN False MODEL.SimrelRCNN.DIM_ENT_PRE_CLS 1024 MODEL.SimrelRCNN.DIM_ENT_PRE_REG 256 MODEL.SimrelRCNN.ONE_REL_CONV True MODEL.SimrelRCNN.ENABLE_BATCH_REDUCTION True MODEL.SimrelRCNN.USE_HARD_LABEL_KLMATCH True MODEL.SimrelRCNN.DISABLE_OBJ2REL_LOSS True
```

>📋  Before testing, create a new directory, named 'eva_oiv6_real025kl_newe2rposition_fullobjbranch_300q_smalldyconv_prequeryaug9_norel_hardlabel', in ./SGGbench/Outputs/ .

To evaluate my model with our LA, run:

```eval
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 10020 --nproc_per_node=8 tools/relation_test_net.py --config-file "configs/oiv6_simrel_e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 8 SOLVER.VAL_PERIOD 12000 SOLVER.MAX_ITER 80000 SOLVER.STEPS '(47000, 64000)' SOLVER.BASE_LR 0.000008 SOLVER.OPTIMIZER "ADAMW" MODEL.WEIGHT Outputs/oiv6_real025kl_newe2rposition_fullobjbranch_300q_smalldyconv_prequeryaug9_norel_hardlabel/model_final.pth OUTPUT_DIR Outputs/la_oiv6_real025kl_newe2rposition_fullobjbranch_300q_smalldyconv_prequeryaug9_norel_hardlabel MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM 1024 MODEL.SimrelRCNN.ENABLE_BG_OBJ False MODEL.SimrelRCNN.ENABLE_REL_X2Y True MODEL.SimrelRCNN.REL_DIM 256 MODEL.SimrelRCNN.NUM_PROPOSALS 300 MODEL.SimrelRCNN.NUM_HEADS 6 MODEL.SimrelRCNN.REL_STACK_NUM 6 MODEL.SimrelRCNN.TRIPLET_MASK_WEIGHT 1.0 MODEL.SimrelRCNN.FREEZE_BACKBONE True MODEL.SimrelRCNN.CROSS_OBJ_FEAT_FUSION False MODEL.SimrelRCNN.CLASS_WEIGHT 1.333 MODEL.SimrelRCNN.L1_WEIGHT 5.0 MODEL.SimrelRCNN.GIOU_WEIGHT 2.0 GLOVE_DIR '' MODEL.SimrelRCNN.ENABLE_FREQ False MODEL.SimrelRCNN.ENABLE_QUERY_REVERSE False MODEL.SimrelRCNN.USE_REFINE_OBJ_FEATURE True MODEL.SimrelRCNN.FREEZE_PUREE_OBJDET False MODEL.SimrelRCNN.PURE_ENT_NUM_PROPOSALS 100 MODEL.SimrelRCNN.ENABLE_MASK_BRANCH False MODEL.SimrelRCNN.KL_BRANCH_WEIGHT 0.25 MODEL.SimrelRCNN.ENABLE_KL_BRANCH True MODEL.SimrelRCNN.POSI_ENCODE_DIM 64 MODEL.SimrelRCNN.REL_CLASS_WEIGHT 1.334 MODEL.SimrelRCNN.PURE_ENT_CLASS_WEIGHT 2.0 MODEL.SimrelRCNN.PURE_ENT_GIOU_WEIGHT 2.0 MODEL.SimrelRCNN.PURE_ENT_L1_WEIGHT 5.0 MODEL.SimrelRCNN.AUXILIARY_BRANCH True MODEL.SimrelRCNN.AUXILIARY_BRANCH_SELECT_ENT_MAX_NUM 25 MODEL.SimrelRCNN.AUXILIARY_BRANCH_START 11 MODEL.SimrelRCNN.ENABLE_ENT_PROP True MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS False MODEL.SimrelRCNN.USE_CROSS_RANK False MODEL.SimrelRCNN.DISABLE_KQ_FUSION_SELFATTEN False MODEL.SimrelRCNN.DIM_ENT_PRE_CLS 1024 MODEL.SimrelRCNN.DIM_ENT_PRE_REG 256 MODEL.SimrelRCNN.ONE_REL_CONV True MODEL.SimrelRCNN.ENABLE_BATCH_REDUCTION True MODEL.SimrelRCNN.USE_HARD_LABEL_KLMATCH True MODEL.SimrelRCNN.DISABLE_OBJ2REL_LOSS True MODEL.SimrelRCNN.REL_LOGITS_ADJUSTMENT True MODEL.SimrelRCNN.LOGIT_ADJ_TAU 0.3
```

>📋  Before testing, create a new directory, named 'la_oiv6_real025kl_newe2rposition_fullobjbranch_300q_smalldyconv_prequeryaug9_norel_hardlabel', in ./SGGbench/Outputs/ .

## Pre-trained Models

You can download pretrained models here:

### VG

#### 300 queries

[Our model](https://drive.google.com/drive/folders/1xZA5VcN9MfOM0-fSnJsHf1fGgmJf7u4Q?usp=sharing) trained on VG dataset. The details for training are presented in our paper. Put the file into './Outputs/1024_8bs_300q_100_more_abla_10/' for testing.



### OI V4

#### 300 queries



### OI V6

#### 300 queries



## Results

Our model achieves the following performance on :

### Visual Genome

\* means 800 queries.

| Models          | SGGen R@20 | SGGen R@50 | SGGen R@100 | SGGen zR@20 | SGGen zR@50 | SGGen zR@100 | SGGen mR@20 | SGGen mR@50 | SGGen mR@100 |
| --------------- | ---------- | ---------- | ----------- | ----------- | ----------- | ------------ | ----------- | ----------- | ------------ |
| Our model       | 25.8      |  32.7       |  36.9    | 1.5     |2.7       |  3.7        | 6.1      | 8.4         | 10.0          |
| Our model\*     | 26.1       | 33.5       | 38.4        | 1.5         | 2.7        | 4.0      | 6.2         | 8.6         | 10.3         |
| Our model+TDE   | 14.5     | 18.3      | 21.0     | 1.8         | 2.7         | 3.6        | 10.8     | 15.0        | 18.5         |
| Our model\*+TDE   | 15.0     | 19.7      | 22.9    | 1.6         | 2.7         | 3.8      | 9.8     | 14.6        | 18.0        |
| Our model+LA    | 18.4    | 23.3     | 26.5     | 1.9      | 2.9         | 4.0       | 13.5       | 17.9       | 21.4        |
| Our model\*+LA    | 18.2   | 23.7    | 27.3     | 2.0     | 3.1       | 4.5     | 13.7        | 18.6       | 22.5        |

## Acknowledgement

Our code is mainly based on: [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), [SparseR-CNN](https://github.com/PeizeSun/SparseR-CNN) and [BGNN-SGG](https://github.com/SHTUPLUS/PySGG).

For this paper, I'm extremely grateful to my advisor [Prof. Limin Wang](https://scholar.google.com/citations?user=HEuN8PcAAAAJ&hl=zh-CN&oi=ao). I should also appreciate my group members for discussing with me: [Jing Tan](https://github.com/sparkstj), [Ziteng Gao](https://github.com/sebgao) and [Jiaqi Tang](https://github.com/tony2016uestc).

## Citations
