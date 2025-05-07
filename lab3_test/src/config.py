
"""
Global configuration for Cascade Mask R‑CNN instance‑segmentation project.
"""

import pathlib

DATA_ROOT = pathlib.Path("./dataset")

NUM_CLASSES = 4
CLASS_NAME_MAP = {i: f"class{i}" for i in range(1, NUM_CLASSES+1)}

VAL_RATIO = 0.2
RANDOM_SEED = 42

# CASCADE_YAML = "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
CASCADE_YAML = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

OUTPUT_DIR = pathlib.Path("./output")
