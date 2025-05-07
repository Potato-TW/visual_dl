
"""
Global configuration for Cascade Mask R‑CNN instance‑segmentation project.
"""

import pathlib

DATA_ROOT = pathlib.Path("./dataset")

NUM_CLASSES = 4
CLASS_NAME_MAP = {}

VAL_RATIO = 0.2
SEED = 123

MODEL_YAML = "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
# MODEL_YAML = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

OUTPUT_DIR = pathlib.Path("./log")

NUM_WORKERS = 8

DATASET_NAME = "dataset"

TOTAL_ITER = 20000