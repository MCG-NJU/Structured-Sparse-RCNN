# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the maximum image side during training will be
# INPUT.MAX_SIZE_TRAIN, while for testing it will be
# INPUT.MAX_SIZE_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.FLIP_AUG = False
_C.MODEL.RPN_ONLY = False
_C.MODEL.MASK_ON = False
_C.MODEL.RETINANET_ON = False
_C.MODEL.KEYPOINT_ON = False
_C.MODEL.ATTRIBUTE_ON = False
_C.MODEL.RELATION_ON = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
_C.MODEL.CLS_AGNOSTIC_BBOX_REG = False

# If the WEIGHT starts with a catalog://, like :R-50, the code will look for
# the path in paths_catalog. Else, it will use it as the specified absolute
# path
_C.MODEL.WEIGHT = ""

# checkpoint of detector, for relation prediction
_C.MODEL.PRETRAINED_DETECTOR_CKPT = ""

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = (800,)  # (800,)
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1333
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [1., 1., 1.]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.TO_BGR255 = True

# Image ColorJitter
_C.INPUT.BRIGHTNESS = 0.0
_C.INPUT.CONTRAST = 0.0
_C.INPUT.SATURATION = 0.0
_C.INPUT.HUE = 0.0

_C.INPUT.VERTICAL_FLIP_PROB_TRAIN = 0.0

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for val, as present in paths_catalog.py
# Note that except dataset names, all remaining val configs reuse those of test
_C.DATASETS.VAL = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 0
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = True


# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
# (e.g., 'FPN.add_fpn_ResNet101_conv5_body' to specify a ResNet-101-FPN
# backbone)
_C.MODEL.BACKBONE.CONV_BODY = "R-50-C4"

# Add StopGrad at a specified stage so the bottom layers are frozen
_C.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 2


# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.FPN = CN()
_C.MODEL.FPN.USE_GN = False
_C.MODEL.FPN.USE_RELU = False


# ---------------------------------------------------------------------------- #
# Group Norm options
# ---------------------------------------------------------------------------- #
_C.MODEL.GROUP_NORM = CN()
# Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
_C.MODEL.GROUP_NORM.DIM_PER_GP = -1
# Number of groups in GroupNorm (-1 if using DIM_PER_GP)
_C.MODEL.GROUP_NORM.NUM_GROUPS = 32
# GroupNorm's small constant in the denominator
_C.MODEL.GROUP_NORM.EPSILON = 1e-5


# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN()
_C.MODEL.RPN.USE_FPN = False
_C.MODEL.RPN.RPN_MID_CHANNEL = 512
# Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
_C.MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
# Stride of the feature map that RPN is attached.
# For FPN, number of strides should match number of scales
_C.MODEL.RPN.ANCHOR_STRIDE = (16,)
# RPN anchor aspect ratios
_C.MODEL.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
# Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
_C.MODEL.RPN.STRADDLE_THRESH = 0
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example)
_C.MODEL.RPN.FG_IOU_THRESHOLD = 0.7
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example)
_C.MODEL.RPN.BG_IOU_THRESHOLD = 0.3
# Total number of RPN examples per image
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of foreground (positive) examples per RPN minibatch
_C.MODEL.RPN.POSITIVE_FRACTION = 0.5
# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
_C.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOP_N_TEST = 6000
# Number of top scoring RPN proposals to keep after applying NMS
_C.MODEL.RPN.POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOP_N_TEST = 1000
# NMS threshold used on RPN proposals
_C.MODEL.RPN.NMS_THRESH = 0.7
# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (a the scale used during training or inference)
_C.MODEL.RPN.MIN_SIZE = 0
# Number of top scoring RPN proposals to keep after combining proposals from
# all FPN levels
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 2000
# Apply the post NMS per batch (default) or per image during training
# (default is True to be consistent with Detectron, see Issue #672)
_C.MODEL.RPN.FPN_POST_NMS_PER_BATCH = True
# Custom rpn head, empty to use default conv or separable conv
_C.MODEL.RPN.RPN_HEAD = "SingleConvRPNHead"


# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_HEADS = CN()
_C.MODEL.ROI_HEADS.USE_FPN = False
# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
_C.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.5
# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
_C.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.3

# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
_C.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)
# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH
# E.g., a common configuration is: 512 * 2 * 8 = 8192
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
_C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

# Only used on test mode

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
_C.MODEL.ROI_HEADS.SCORE_THRESH = 0.01
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
_C.MODEL.ROI_HEADS.NMS = 0.3
_C.MODEL.ROI_HEADS.POST_NMS_PER_CLS_TOPN = 300
# Remove duplicated assigned labels for a single bbox in nms
_C.MODEL.ROI_HEADS.NMS_FILTER_DUPLICATES = False 
# Maximum number of detections to return per image (100 is based on the limit
# established for the COCO dataset)
_C.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 256


_C.MODEL.ROI_BOX_HEAD = CN()
_C.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_BOX_HEAD.PREDICTOR = "FastRCNNPredictor"
_C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 81
# Hidden layer dimension when using an MLP for the RoI box head
_C.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 2048
# GN
_C.MODEL.ROI_BOX_HEAD.USE_GN = False
# Dilation
_C.MODEL.ROI_BOX_HEAD.DILATION = 1
_C.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM = 256
_C.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS = 4



_C.MODEL.ROI_ATTRIBUTE_HEAD = CN()
_C.MODEL.ROI_ATTRIBUTE_HEAD.FEATURE_EXTRACTOR = "FPN2MLPFeatureExtractor"
_C.MODEL.ROI_ATTRIBUTE_HEAD.PREDICTOR = "FPNPredictor"
_C.MODEL.ROI_ATTRIBUTE_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True
# Add attributes to each box
_C.MODEL.ROI_ATTRIBUTE_HEAD.USE_BINARY_LOSS = True
_C.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_LOSS_WEIGHT = 0.1
_C.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES = 201
_C.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES = 10  # max number of attribute per bbox
_C.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE = True
_C.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO = 3
_C.MODEL.ROI_ATTRIBUTE_HEAD.POS_WEIGHT = 5.0


_C.MODEL.ROI_MASK_HEAD = CN()
_C.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_MASK_HEAD.PREDICTOR = "MaskRCNNC4Predictor"
_C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_MASK_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_MASK_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_MASK_HEAD.CONV_LAYERS = (256, 256, 256, 256)
_C.MODEL.ROI_MASK_HEAD.RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True
# Whether or not resize and translate masks to the input image.
_C.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS = False
_C.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD = 0.5
# Dilation
_C.MODEL.ROI_MASK_HEAD.DILATION = 1
# GN
_C.MODEL.ROI_MASK_HEAD.USE_GN = False

_C.MODEL.ROI_KEYPOINT_HEAD = CN()
_C.MODEL.ROI_KEYPOINT_HEAD.FEATURE_EXTRACTOR = "KeypointRCNNFeatureExtractor"
_C.MODEL.ROI_KEYPOINT_HEAD.PREDICTOR = "KeypointRCNNPredictor"
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_KEYPOINT_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_KEYPOINT_HEAD.CONV_LAYERS = tuple(512 for _ in range(8))
_C.MODEL.ROI_KEYPOINT_HEAD.RESOLUTION = 14
_C.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES = 17
_C.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True


_C.MODEL.ROI_RELATION_HEAD = CN()
# share box feature extractor should be set False for neural-motifs
_C.MODEL.ROI_RELATION_HEAD.PREDICTOR = "MotifPredictor"
_C.MODEL.ROI_RELATION_HEAD.FEATURE_EXTRACTOR = "RelationFeatureExtractor"
_C.MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS = True
_C.MODEL.ROI_RELATION_HEAD.NUM_CLASSES = 51
_C.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE = 64
_C.MODEL.ROI_RELATION_HEAD.POSITIVE_FRACTION = 0.25
_C.MODEL.ROI_RELATION_HEAD.USE_GT_BOX = True
_C.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = False
_C.MODEL.ROI_RELATION_HEAD.EMBED_DIM = 200
_C.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE = 0.2
_C.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM = 512
_C.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM = 4096
_C.MODEL.ROI_RELATION_HEAD.CONTEXT_OBJ_LAYER = 1  # assert >= 1
_C.MODEL.ROI_RELATION_HEAD.CONTEXT_REL_LAYER = 1  # assert >= 1

_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER = CN()
# for TransformerPredictor only
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE = 0.1   
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER = 4        
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER = 2        
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD = 8         
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM = 2048     
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM = 64         
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM = 64         

_C.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS = False
_C.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION = True
_C.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS = True
_C.MODEL.ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP = True
_C.MODEL.ROI_RELATION_HEAD.NUM_SAMPLE_PER_GT_REL = 4  # when sample fg relationship from gt, the max number of corresponding proposal pairs

# in sgdet, to make sure the detector won't missing any ground truth bbox, 
# we add grount truth box to the output of RPN proposals during Training
_C.MODEL.ROI_RELATION_HEAD.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN = False


_C.MODEL.ROI_RELATION_HEAD.CAUSAL = CN()
# direct and indirect effect analysis
_C.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS = False
# Fusion
_C.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE = 'sum'
# causal context feature layer
_C.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER = 'motifs'
# separate spatial in union feature
_C.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL = False

_C.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION = False

_C.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE = 'none' # 'TDE', 'TIE', 'TE'

# proportion of predicates
_C.MODEL.ROI_RELATION_HEAD.REL_PROP = [0.01858, 0.00057, 0.00051, 0.00109, 0.00150, 0.00489, 0.00432, 0.02913, 0.00245, 
                                        0.00121, 0.00404, 0.00110, 0.00132, 0.00172, 0.00005, 0.00242, 0.00050, 0.00048, 
                                        0.00208, 0.15608, 0.02650, 0.06091, 0.00900, 0.00183, 0.00225, 0.00090, 0.00028, 
                                        0.00077, 0.04844, 0.08645, 0.31621, 0.00088, 0.00301, 0.00042, 0.00186, 0.00100, 
                                        0.00027, 0.01012, 0.00010, 0.01286, 0.00647, 0.00084, 0.01077, 0.00132, 0.00069, 
                                        0.00376, 0.00214, 0.11424, 0.01205, 0.02958]


_C.MODEL.ROI_RELATION_HEAD.LONGTAIL_PART_DICT = [None, 'b', 't', 't', 't', 't', 't', 't', 'b', 't', 't', 't', 't', 't',
                                                 't', 't', 't', 't', 't', 't', 'h', 'b', 'b', 'b', 't', 't', 't', 't',
                                                 't', 'b', 'h', 'h', 't', 't', 't', 't', 't', 't', 'b', 't', 'b', 'b',
                                                 't', 'b', 't', 't', 't', 't', 'h', 'b', 'b']


"""
Add config for SimrelRCNN (from SparseRCNN).
"""
_C.MODEL.SimrelRCNN = CN()
_C.MODEL.SimrelRCNN.CROSS_OBJ_FEAT_FUSION = False

_C.MODEL.SimrelRCNN.TRAIN_OBJ = False
_C.MODEL.SimrelRCNN.USE_HUNGARIAN_LOSS = False
_C.MODEL.SimrelRCNN.USE_RELATION_FUSION_FOR_OBJECT = False

_C.MODEL.SimrelRCNN.NUM_CLASSES = 151  #80
_C.MODEL.SimrelRCNN.NUM_PROPOSALS = 150 #300


_C.MODEL.SimrelRCNN.PURE_ENT_NUM_PROPOSALS = 150 #300

# RCNN Head.
_C.MODEL.SimrelRCNN.NHEADS = 8
_C.MODEL.SimrelRCNN.DROPOUT = 0.0
_C.MODEL.SimrelRCNN.DIM_FEEDFORWARD = 2048
_C.MODEL.SimrelRCNN.ACTIVATION = 'relu'
_C.MODEL.SimrelRCNN.HIDDEN_DIM = 256 #dec
_C.MODEL.SimrelRCNN.NUM_CLS = 1
_C.MODEL.SimrelRCNN.NUM_REG = 3
_C.MODEL.SimrelRCNN.NUM_HEADS = 6

_C.MODEL.SimrelRCNN.NUM_CLS_REL = 1
_C.MODEL.SimrelRCNN.SIAMESE_HEAD = True

_C.MODEL.SimrelRCNN.ENABLE_REL_X2Y = False
_C.MODEL.SimrelRCNN.ENABLE_FREQ = False

_C.MODEL.SimrelRCNN.POSI_ENCODE_DIM = 64
_C.MODEL.SimrelRCNN.POSI_EMBED_DIM = 256
_C.MODEL.SimrelRCNN.REL_STACK_NUM = 6 #4
_C.MODEL.SimrelRCNN.REL_DIM = 512

# Dynamic Conv.
_C.MODEL.SimrelRCNN.NUM_DYNAMIC = 2
_C.MODEL.SimrelRCNN.DIM_DYNAMIC = 64

# Loss.
_C.MODEL.SimrelRCNN.CLASS_WEIGHT = 2.0 #2.0
_C.MODEL.SimrelRCNN.GIOU_WEIGHT = 2.0 #4.0 #2.0
_C.MODEL.SimrelRCNN.L1_WEIGHT = 5.0 #10.0 #5.0
_C.MODEL.SimrelRCNN.DEEP_SUPERVISION = True
_C.MODEL.SimrelRCNN.NO_OBJECT_WEIGHT = 0.1

_C.MODEL.SimrelRCNN.REL_CLASS_WEIGHT = 4.0 #4.0
_C.MODEL.SimrelRCNN.TRIPLET_MASK_WEIGHT = 1.0

# Focal Loss.
_C.MODEL.SimrelRCNN.USE_FOCAL = True
_C.MODEL.SimrelRCNN.ALPHA = 0.25
_C.MODEL.SimrelRCNN.GAMMA = 2.0
_C.MODEL.SimrelRCNN.PRIOR_PROB = 0.01

_C.MODEL.SimrelRCNN.PRIOR_PROB_REL = 0.01



_C.MODEL.SimrelRCNN.USE_EQU_LOSS = False
_C.MODEL.SimrelRCNN.FREEZE_BACKBONE = False

_C.MODEL.SimrelRCNN.FREEZE_PUREE_OBJDET = False

_C.MODEL.SimrelRCNN.ENABLE_BG_OBJ = True

_C.MODEL.SimrelRCNN.ENABLE_MASK_BRANCH = False

_C.MODEL.SimrelRCNN.ENABLE_QUERY_REVERSE = False
_C.MODEL.SimrelRCNN.USE_REFINE_OBJ_FEATURE = False

_C.MODEL.SimrelRCNN.PURE_ENT_CLASS_WEIGHT = 2.0
_C.MODEL.SimrelRCNN.PURE_ENT_GIOU_WEIGHT = 2.0
_C.MODEL.SimrelRCNN.PURE_ENT_L1_WEIGHT = 5.0


_C.MODEL.SimrelRCNN.ENABLE_FAKE_TRUE_LABEL = False

_C.MODEL.SimrelRCNN.ENABLE_KL_BRANCH = False
_C.MODEL.SimrelRCNN.KL_BRANCH_WEIGHT = 0.5

_C.MODEL.SimrelRCNN.AUXILIARY_BRANCH = False
_C.MODEL.SimrelRCNN.AUXILIARY_BRANCH_SELECT_ENT_MAX_NUM = 2
_C.MODEL.SimrelRCNN.AUXILIARY_BRANCH_START = 0

_C.MODEL.SimrelRCNN.PAIR_GROUP = 5

_C.MODEL.SimrelRCNN.LABEL_SMOOTHING_EPS = -1.


_C.MODEL.SimrelRCNN.VRD_ENT_PROP = [0.002948, 0.008961, 0.002075, 0.002685, 0.00112, 0.003624, 0.006523, 0.010031, \
0.008565, 0.005963, 0.002339, 0.004052, \
0.037819, 0.02194, 0.002833, 0.001977, 0.002421, 0.001482, 0.026091, 0.002438, 0.003196, 0.012041, 0.00677, 0.004349, \
0.001993, 0.002668, 0.004003, 0.004381, 0.006803, 0.00392, 0.002652, 0.001219, 0.001367, 0.001499, 0.001532, 0.015368, \
0.010855, 0.002668, 0.017542, 0.007017, 0.00448, 0.001664, 0.01405, 0.006457, 0.00364, 0.003657, 0.00481, 0.007412, \
0.001828, 0.008483, 0.010525, 0.005864, 0.002405, 0.00084, 0.017509, 0.001515, 0.290166, 0.005979, 0.00336, 0.003443, \
0.003937, 0.002635, 0.007626, 0.003673, 0.002306, 0.001598, 0.001186, 0.011662, 0.00509, 0.002685, 0.002323, 0.035513, \
0.001861, 0.009109, 0.006029, 0.002817, 0.006408, 0.002767, 0.049942, 0.002141, 0.004332, 0.001598, 0.039236, 0.001515, \
0.00509, 0.002355, 0.024148, 0.002899, 0.006375, 0.002273, 0.007511, 0.002108, 0.016422, 0.01153, 0.01405, 0.011069, \
0.004596, 0.001861, 0.00229, 0.011613]

#_C.MODEL.SimrelRCNN.VRD_ENT_PROP = [0.290166, 0.049942, 0.037819, 0.01405, 0.02194, 0.024148, 0.035513, \
#0.012041, 0.026091, 0.007511, 0.015368, 0.016422, \
#0.008565, 0.017542, 0.01153, 0.010855, 0.017509, 0.011662, 0.010525, 0.01405, 0.008483, 0.011613, 0.011069, 0.007626, \
#0.010031, 0.00677, 0.008961, 0.001861, 0.007412, 0.006803, 0.001977, 0.004003, 0.006523, 0.009109, 0.006375, 0.005963, \
#0.007017, 0.001598, 0.00481, 0.004349, 0.003624, 0.00392, 0.005864, 0.00448, 0.003937, 0.00509, 0.006408, 0.002273, \
#0.002833, 0.005979, 0.002948, 0.004332, 0.004381, 0.002817, 0.002323, 0.004052, 0.004596, 0.002668, 0.006029, 0.003673, \
#0.006457, 0.003196, 0.00509, 0.002339, 0.001993, 0.00336, 0.003443, 0.002685, 0.002652, 0.003657, 0.002685, 0.00364, \
#0.002635, 0.001482, 0.001861, 0.001186, 0.002438, 0.002767, 0.002306, 0.002355, 0.001515, 0.002405, 0.002108, 0.002668, \
#0.002421, 0.002075, 0.00112, 0.001532, 0.002899, 0.001828, 0.001499, 0.001664, 0.002141, 0.00084, 0.001219, 0.00229, \
#0.001367, 0.039236, 0.001598, 0.001515]

_C.MODEL.SimrelRCNN.OI_REL_04_ENT_PROP = [0.000434, 0.009658, 6.2e-05, 0.000214, 0.002714, 0.004314, \
    0.004507, 0.000229, 0.201548, 0.002618, 5.6e-05, 0.004328, 0.00086, 0.001215, 0.007706, 0.000428, \
    0.006013, 0.000828, 0.011603, 5e-05, 0.000264, 0.107005, 0.002694, 3e-06, 0.000522, 0.025187, \
    5.6e-05, 2.9e-05, 0.011776, 0.000493, 0.000414, 0.284859, 1.2e-05, 0.001655, 8.8e-05, 0.006344, \
    0.179668, 0.017217, 0.002665, 0.000616, 0.006937, 0.033049, 0.027908, 0.000355, 0.016014, 0.006183, \
    0.001538, 0.000135, 0.00039, 0.001643, 8.8e-05, 0.00017, 0.000863, 0.003017, 0.000742, 3e-06, 1.2e-05]
    
_C.MODEL.SimrelRCNN.OI_REL_06_ENT_PROP = [3e-05, 1e-08, 1e-08, 1.2e-05, 0.002774, 1e-08, 0.0001, \
    1e-08, 1e-08, 1e-08, 1e-08, 2.4e-05, 1e-08, 0.000523, 1e-08, 1e-08, 0.000972, 1e-08, 0.003973, \
    6.7e-05, 1e-08, 1e-08, 1e-08, 1e-08, 1e-06, 3e-05, 1.8e-05, 1e-08, 0.000177, 1e-08, 7e-06, 1e-08, \
    1e-08, 0.00019, 0.028317, 0.000177, 0.000603, 1e-08, 0.000262, 0.009336, 1e-08, 1e-08, 0.004818, \
    0.000797, 0.001197, 1e-08, 1e-08, 1e-08, 1e-08, 0.003257, 0.011662, 1e-08, 0.000299, 1e-08, 1e-08, \
    1e-08, 1e-08, 1e-08, 1e-08, 1.2e-05, 5e-06, 2e-06, 1e-08, 1e-08, 7e-06, 5e-06, 1e-08, 1e-05, 1e-08, \
    5.7e-05, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-05, 2e-06, 1e-08, 0.000906, 1e-08, 0.002782, \
    1e-08, 4.8e-05, 1e-08, 1e-08, 1e-08, 0.000473, 8.6e-05, 0.002101, 1e-08, 1e-06, 1e-08, 1e-08, 1e-08, \
    1e-08, 0.096224, 1e-08, 1e-08, 0.000107, 1e-08, 1e-08, 1e-08, 0.000317, 1e-08, 1e-08, 1e-08, 0.000121, \
    0.000154, 1e-08, 1e-08, 5e-06, 1e-08, 1e-08, 1e-08, 0.001098, 3.1e-05, 1e-08, 1e-08, 1e-08, 0.014905, \
    0.001309, 1e-08, 0.000295, 9.6e-05, 1e-08, 2.8e-05, 1e-08, 3e-05, 1e-08, 1e-08, 1e-08, 1e-08, 0.001233, \
    1e-08, 7.6e-05, 1e-08, 3.5e-05, 0.002688, 0.000794, 9.6e-05, 0.004375, 1e-08, 1e-08, 1e-08, 8e-06, 1e-08, \
    1e-08, 1e-08, 1e-08, 1e-06, 1.2e-05, 1e-08, 1e-08, 1.1e-05, 1e-08, 1e-08, 2.2e-05, 1e-08, 1e-08, 1e-08, \
    1e-08, 1e-08, 0.003901, 1e-08, 1e-08, 0.002602, 2.8e-05, 0.000387, 5.3e-05, 0.001078, 1e-08, 1e-06, 1e-05, \
    1e-08, 0.009539, 1e-08, 0.001628, 0.004885, 0.000229, 1e-08, 1e-08, 1e-08, 1e-08, 3e-05, 1e-08, 1e-08, 1e-08, \
    0.002643, 1e-06, 1e-08, 1e-08, 1e-08, 0.006427, 1e-05, 5.5e-05, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, \
    1e-08, 5e-06, 1e-08, 8e-06, 1.2e-05, 2.5e-05, 1e-06, 1e-06, 1e-08, 0.000522, 1e-08, 1e-08, 0.000143, 1e-08, \
    1.5e-05, 1e-08, 0.017786, 1e-08, 1e-08, 1e-08, 0.000124, 1e-08, 1e-08, 1.5e-05, 1e-08, 0.122525, 1e-08, \
    1e-08, 1e-08, 1e-08, 0.015028, 1e-08, 0.001173, 0.002785, 3.4e-05, 1e-08, 1e-08, 1e-08, 1e-08, 0.004464, \
    1.7e-05, 5e-06, 1e-08, 0.000467, 1e-06, 2e-05, 0.001174, 0.003495, 2.2e-05, 1.5e-05, 1e-08, 1e-08, 1e-08, \
    1.4e-05, 1e-08, 0.000852, 1e-08, 1e-08, 1e-08, 7e-06, 1e-08, 1e-08, 1e-08, 1.1e-05, 1e-08, 1e-08, 1e-08, \
    1e-08, 1e-08, 2e-06, 2.8e-05, 1e-08, 8e-06, 1e-06, 1e-08, 1e-08, 1e-08, 1e-06, 0.152091, 1e-08, 1e-08, \
    1e-08, 2.4e-05, 0.006122, 1e-08, 1e-06, 1e-05, 1e-08, 5e-06, 1e-08, 0.000236, 1e-08, 1e-08, 2.1e-05, \
    0.000249, 1e-08, 1e-08, 0.00431, 1e-08, 0.000278, 5e-06, 1e-06, 1e-08, 1e-08, 1e-08, 0.223093, 1e-08, \
    2e-06, 1e-08, 3.4e-05, 0.000875, 1e-08, 1e-06, 1e-08, 1.2e-05, 1e-08, 9.3e-05, 0.000311, 1e-08, 0.00039, \
    1e-08, 8e-06, 6.8e-05, 1e-08, 5.5e-05, 8.3e-05, 4.4e-05, 1e-08, 1e-08, 0.000223, 0.032137, 1e-08, 1e-05, \
    1e-08, 1e-08, 5e-06, 1e-08, 0.000103, 4.3e-05, 0.0001, 4e-06, 1e-08, 1e-08, 1e-08, 5.5e-05, 1e-08, 1e-08, \
    1e-08, 0.000477, 1e-08, 0.000573, 1e-08, 0.001906, 1e-08, 0.000176, 1e-08, 1e-08, 0.000681, 9.3e-05, 1e-08, \
    5e-06, 7.8e-05, 1.2e-05, 1e-08, 0.000126, 8.4e-05, 0.000295, 1e-08, 0.000176, 1e-06, 1e-08, 1e-08, 4.4e-05, \
    1e-08, 9.4e-05, 0.008517, 1e-08, 1e-08, 1e-08, 0.000352, 0.001424, 0.000273, 4e-06, 1e-08, 4e-06, 1e-08, \
    0.001375, 0.000978, 0.001689, 0.00012, 0.000672, 1e-08, 6.1e-05, 1e-08, 1e-08, 7.8e-05, 0.004182, 0.000883, \
    1e-08, 1e-08, 0.001649, 1e-08, 0.001468, 1e-08, 0.005459, 1e-08, 1e-08, 1e-08, 1.2e-05, 2e-06, 1e-08, 1e-08, \
    3.2e-05, 1e-08, 1e-08, 5.4e-05, 1e-08, 5.3e-05, 0.000793, 1e-08, 1e-08, 4.7e-05, 1e-08, 1e-08, 1e-08, 1e-08, \
    3.7e-05, 1e-08, 1e-06, 1e-08, 1e-08, 1e-08, 0.000216, 1e-08, 1e-08, 1e-08, 6.1e-05, 1e-08, 0.012293, 9.4e-05, \
    0.000104, 1.2e-05, 1e-08, 0.005502, 0.000229, 0.000394, 8e-06, 0.001509, 1e-08, 2e-06, 1e-08, 1e-08, 1e-08, \
    1e-08, 1e-08, 1e-08, 0.000233, 1.8e-05, 1.1e-05, 1e-08, 1e-08, 5e-05, 1e-06, 1e-08, 1e-08, 0.000245, 0.000232, \
    1e-08, 1e-08, 2.5e-05, 8e-06, 1e-08, 1e-08, 0.001374, 0.000235, 1e-08, 1e-08, 1.8e-05, 0.000834, 1e-08, 1e-08, \
    1e-08, 1e-08, 1e-06, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 5.3e-05, 0.000203, 1e-08, 1e-08, 1e-08, 0.00039, 1e-08, \
    0.002468, 1e-08, 0.000407, 1e-08, 1e-08, 1e-08, 1e-08, 2.4e-05, 1e-08, 1e-08, 1e-08, 1e-08, 5e-06, 2e-05, 4e-05, \
    1e-08, 0.015377, 3.1e-05, 4e-06, 7.6e-05, 1e-08, 0.002953, 1e-08, 4.7e-05, 1e-06, 7e-06, 1e-08, 4e-06, 1e-08, \
    1e-08, 0.000186, 4.4e-05, 0.000139, 1e-08, 6e-05, 1e-08, 0.000225, 1e-08, 1e-08, 1e-08, 1e-06, 1e-08, 0.000634, \
    1e-08, 5e-06, 5e-06, 0.00019, 0.000285, 0.000101, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 0.00903, \
    2.2e-05, 1e-08, 1e-08, 9.6e-05, 1e-08, 1e-08, 8e-06, 1e-08, 1.4e-05, 2.4e-05, 1.7e-05, 0.072069, 1e-08, 2.4e-05, \
    1e-08, 0.006698, 1e-08, 1e-08, 2e-06, 0.000312, 6e-05, 2e-06, 1e-08, 0.000251, 5.7e-05, 1e-08, 4.3e-05, 1e-06, \
    1.8e-05, 2e-06, 1e-08, 0.000582, 1e-08, 4.5e-05, 0.000457, 1e-06, 1e-08, 2e-05, 6.1e-05, 0.001031, 1e-08, \
    0.000395, 0.000159, 1e-08, 1e-08, 1e-08]


_C.MODEL.SimrelRCNN.VG_ENT_PROP = [0.002447, 0.001386, 0.005004, 0.0043, 0.00265, 0.001761, 0.006333, \
    0.006127, 0.006696, 0.006918, 0.007002, 0.005286, 0.002759, 0.00363, 0.003416, 0.001529, 0.003506, \
    0.003663, 0.002824, 0.012767, 0.003485, 0.031631, 0.010871, 0.001816, 0.002957, 0.011522, 0.010673, \
    0.00777, 0.003512, 0.006633, 0.002885, 0.003086, 0.004577, 0.002561, 0.001654, 0.003154, 0.009541, \
    0.007835, 0.001017, 0.010561, 0.010074, 0.002526, 0.004707, 0.005647, 0.007415, 0.001564, 0.001874, \
    0.005126, 0.005519, 0.002634, 0.001607, 0.009931, 0.00947, 0.009544, 0.002797, 0.001606, 0.012209, \
    0.011531, 0.003544, 0.008968, 0.016847, 0.005578, 0.002674, 0.008491, 0.00353, 0.008128, 0.004575, \
    0.001784, 0.001683, 0.001808, 0.002136, 0.004286, 0.006867, 0.015155, 0.002653, 0.005762, 0.002756, \
    0.093659, 0.000969, 0.005911, 0.002627, 0.002465, 0.002805, 0.005279, 0.00194, 0.002335, 0.008317, \
    0.002366, 0.002024, 0.006674, 0.029837, 0.003273, 0.003819, 0.004176, 0.008474, 0.003267, 0.011468, \
    0.003497, 0.011084, 0.002454, 0.001986, 0.003095, 0.001077, 0.003041, 0.003312, 0.002181, 0.001214, \
    0.002633, 0.003119, 0.003826, 0.027624, 0.0065, 0.006525, 0.007574, 0.014045, 0.001871, 0.005691, \
    0.003329, 0.00258, 0.001083, 0.010043, 0.001532, 0.001221, 0.011732, 0.005048, 0.023312, 0.008504, \
    0.002804, 0.000904, 0.003837, 0.002158, 0.001207, 0.002979, 0.007442, 0.015481, 0.024469, 0.00477, \
    0.004053, 0.005952, 0.004289, 0.001138, 0.001498, 0.00183, 0.006507, 0.033803, 0.002553, 0.003998, \
    0.001764, 0.036275, 0.007117]
_C.MODEL.SimrelRCNN.ENABLE_ENT_PROP = False
_C.MODEL.SimrelRCNN.ENT_FREQ_MU = 4.

_C.MODEL.SimrelRCNN.USE_CROSS_RANK = False
_C.MODEL.SimrelRCNN.DISABLE_KQ_FUSION_SELFATTEN = False

_C.MODEL.SimrelRCNN.ENT_DET_ONLY_FG = True

_C.MODEL.SimrelRCNN.REL_LOGITS_ADJUSTMENT = False
_C.MODEL.SimrelRCNN.LOGIT_ADJ_TAU = 0.3
_C.MODEL.SimrelRCNN.USE_ONLY_OBJ2REL = False


_C.MODEL.SimrelRCNN.DISABLE_REL_FUSION = False


_C.MODEL.SimrelRCNN.DIM_ENT_PRE_CLS = None
_C.MODEL.SimrelRCNN.DIM_ENT_PRE_REG = None
_C.MODEL.SimrelRCNN.ONE_REL_CONV = False
_C.MODEL.SimrelRCNN.ENABLE_BATCH_REDUCTION = False
_C.MODEL.SimrelRCNN.NUM_BATCH_REDUCTION = 250

_C.MODEL.SimrelRCNN.FREEZE_LATENT_VECTORS = False
_C.MODEL.SimrelRCNN.FREEZE_LATENT_BOXES = False


_C.MODEL.SimrelRCNN.USE_PURE_OBJDET = False
_C.MODEL.SimrelRCNN.USE_SPARSERCNN_AS_THE_OBJDET_BASE_WHEN_REL = False
_C.MODEL.SimrelRCNN.USE_TRIPLET_NMS = False
_C.MODEL.SimrelRCNN.PAIRS_RANDOM_NUM_FOR_KL = 3000
_C.MODEL.SimrelRCNN.USE_HARD_LABEL_KLMATCH = False
_C.MODEL.SimrelRCNN.USE_LAST_RELNESS = False
_C.MODEL.SimrelRCNN.USE_LAST_DET_FOR_KL_LABELASSIGN = False

_C.MODEL.SimrelRCNN.DISABLE_OBJ2REL_LOSS = False


_C.MODEL.SimrelRCNN.QUERY_GRADUAL_REDUCTION = None


_C.MODEL.VGG = CN()
_C.MODEL.VGG.VGG16_OUT_CHANNELS= 512
# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
_C.MODEL.RESNETS.NUM_GROUPS = 1

# Baseline width of each group
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True

# Residual transformation function
_C.MODEL.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"
# ResNet's stem function (conv1 and pool1)
_C.MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"

# Apply dilation in stage "res5"
_C.MODEL.RESNETS.RES5_DILATION = 1

_C.MODEL.RESNETS.BACKBONE_OUT_CHANNELS = 256 * 4
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

_C.MODEL.RESNETS.STAGE_WITH_DCN = (False, False, False, False)
_C.MODEL.RESNETS.WITH_MODULATED_DCN = False
_C.MODEL.RESNETS.DEFORMABLE_GROUPS = 1


# ---------------------------------------------------------------------------- #
# RetinaNet Options (Follow the Detectron version)
# ---------------------------------------------------------------------------- #
_C.MODEL.RETINANET = CN()

# This is the number of foreground classes and background.
_C.MODEL.RETINANET.NUM_CLASSES = 81

# Anchor aspect ratios to use
_C.MODEL.RETINANET.ANCHOR_SIZES = (32, 64, 128, 256, 512)
_C.MODEL.RETINANET.ASPECT_RATIOS = (0.5, 1.0, 2.0)
_C.MODEL.RETINANET.ANCHOR_STRIDES = (8, 16, 32, 64, 128)
_C.MODEL.RETINANET.STRADDLE_THRESH = 0

# Anchor scales per octave
_C.MODEL.RETINANET.OCTAVE = 2.0
_C.MODEL.RETINANET.SCALES_PER_OCTAVE = 3

# Use C5 or P5 to generate P6
_C.MODEL.RETINANET.USE_C5 = True

# Convolutions to use in the cls and bbox tower
# NOTE: this doesn't include the last conv for logits
_C.MODEL.RETINANET.NUM_CONVS = 4

# Weight for bbox_regression loss
_C.MODEL.RETINANET.BBOX_REG_WEIGHT = 4.0

# Smooth L1 loss beta for bbox regression
_C.MODEL.RETINANET.BBOX_REG_BETA = 0.11

# During inference, #locs to select based on cls score before NMS is performed
# per FPN level
_C.MODEL.RETINANET.PRE_NMS_TOP_N = 1000

# IoU overlap ratio for labeling an anchor as positive
# Anchors with >= iou overlap are labeled positive
_C.MODEL.RETINANET.FG_IOU_THRESHOLD = 0.5

# IoU overlap ratio for labeling an anchor as negative
# Anchors with < iou overlap are labeled negative
_C.MODEL.RETINANET.BG_IOU_THRESHOLD = 0.4

# Focal loss parameter: alpha
_C.MODEL.RETINANET.LOSS_ALPHA = 0.25

# Focal loss parameter: gamma
_C.MODEL.RETINANET.LOSS_GAMMA = 2.0

# Prior prob for the positives at the beginning of training. This is used to set
# the bias init for the logits layer
_C.MODEL.RETINANET.PRIOR_PROB = 0.01

# Inference cls score threshold, anchors with score > INFERENCE_TH are
# considered for inference
_C.MODEL.RETINANET.INFERENCE_TH = 0.05

# NMS threshold used in RetinaNet
_C.MODEL.RETINANET.NMS_TH = 0.4


# ---------------------------------------------------------------------------- #
# FBNet options
# ---------------------------------------------------------------------------- #
_C.MODEL.FBNET = CN()
_C.MODEL.FBNET.ARCH = "default"
# custom arch
_C.MODEL.FBNET.ARCH_DEF = ""
_C.MODEL.FBNET.BN_TYPE = "bn"
_C.MODEL.FBNET.SCALE_FACTOR = 1.0
# the output channels will be divisible by WIDTH_DIVISOR
_C.MODEL.FBNET.WIDTH_DIVISOR = 1
_C.MODEL.FBNET.DW_CONV_SKIP_BN = True
_C.MODEL.FBNET.DW_CONV_SKIP_RELU = True

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.DET_HEAD_LAST_SCALE = 1.0
_C.MODEL.FBNET.DET_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.DET_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.KPTS_HEAD_LAST_SCALE = 0.0
_C.MODEL.FBNET.KPTS_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.KPTS_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.MASK_HEAD_LAST_SCALE = 0.0
_C.MODEL.FBNET.MASK_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.MASK_HEAD_STRIDE = 0

# 0 to use all blocks defined in arch_def
_C.MODEL.FBNET.RPN_HEAD_BLOCKS = 0
_C.MODEL.FBNET.RPN_BN_TYPE = ""


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 0.002
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0
_C.SOLVER.CLIP_NORM = 5.0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.SCHEDULE = CN()
_C.SOLVER.SCHEDULE.TYPE = "WarmupMultiStepLR"  # "WarmupReduceLROnPlateau"
# the following paramters are only used for WarmupReduceLROnPlateau
_C.SOLVER.SCHEDULE.PATIENCE = 2
_C.SOLVER.SCHEDULE.THRESHOLD = 1e-4
_C.SOLVER.SCHEDULE.COOLDOWN = 1
_C.SOLVER.SCHEDULE.FACTOR = 0.5
_C.SOLVER.SCHEDULE.MAX_DECAY_STEP = 111 #7


_C.SOLVER.CHECKPOINT_PERIOD = 2500

_C.SOLVER.GRAD_NORM_CLIP = 5.0

_C.SOLVER.PRINT_GRAD_FREQ = 5000
# whether validate and validate period
_C.SOLVER.TO_VAL = True
_C.SOLVER.PRE_VAL = False #True
_C.SOLVER.VAL_PERIOD = 2500

# update schedule
# when load from a previous model, if set to True
# only maintain the iteration number and all the other settings of the 
# schedule will be changed
_C.SOLVER.UPDATE_SCHEDULE_DURING_LOAD = False

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16

"""
Add Optimizer config for SimrelRCNN (from SparseRCNN).
"""
# Optimizer.
_C.SOLVER.OPTIMIZER = "ADAMW"
_C.SOLVER.BACKBONE_MULTIPLIER = 1.0
_C.SOLVER.CLIP_GRADIENTS = CN()
_C.SOLVER.CLIP_GRADIENTS.ENABLED = True
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "full_model"
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.EXPECTED_RESULTS = []
_C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 8
# Number of detections per image
_C.TEST.DETECTIONS_PER_IMG = 100

# ---------------------------------------------------------------------------- #
# Test-time augmentations for bounding box detection
# See configs/test_time_aug/e2e_mask_rcnn_R-50-FPN_1x.yaml for an example
# ---------------------------------------------------------------------------- #
_C.TEST.BBOX_AUG = CN()

# Enable test-time augmentation for bounding box detection if True
_C.TEST.BBOX_AUG.ENABLED = False

# Horizontal flip at the original scale (id transform)
_C.TEST.BBOX_AUG.H_FLIP = False

# Each scale is the pixel size of an image's shortest side
_C.TEST.BBOX_AUG.SCALES = ()

# Max pixel size of the longer side
_C.TEST.BBOX_AUG.MAX_SIZE = 4000

# Horizontal flip at each scale
_C.TEST.BBOX_AUG.SCALE_H_FLIP = False

_C.TEST.SAVE_PROPOSALS = False
# Settings for relation testing
_C.TEST.RELATION = CN()
_C.TEST.RELATION.MULTIPLE_PREDS = False
_C.TEST.RELATION.IOU_THRESHOLD = 0.5
_C.TEST.RELATION.REQUIRE_OVERLAP = True
# when predict the label of bbox, run nms on each cls
_C.TEST.RELATION.LATER_NMS_PREDICTION_THRES = 0.3 
# synchronize_gather, used for sgdet, otherwise test on multi-gpu will cause out of memory
_C.TEST.RELATION.SYNC_GATHER = False

_C.TEST.ALLOW_LOAD_FROM_CACHE = True


_C.TEST.CUSTUM_EVAL = False
_C.TEST.CUSTUM_PATH = '.'

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
_C.DETECTED_SGG_DIR = "."
_C.GLOVE_DIR = "."

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
_C.PATHS_DATA = os.path.join(os.path.dirname(__file__), "../data/datasets")

# ---------------------------------------------------------------------------- #
# Precision options
# ---------------------------------------------------------------------------- #

# Precision of input, allowable: (float32, float16)
_C.DTYPE = "float32"

# Enable verbosity in apex.amp
_C.AMP_VERBOSE = False


_C.DEBUG = CN()
_C.DEBUG.PURE_SPARSE_RCNN = False
_C.DEBUG.REMOVE_BG_SAMPLE = False
_C.DEBUG.DUPLICATE_OBJ_BOXES = False