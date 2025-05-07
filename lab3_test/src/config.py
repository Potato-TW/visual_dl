import pathlib

DATA_ROOT = pathlib.Path("./dataset")
OUTPUT_DIR = pathlib.Path("./log")
DATASET_NAME = "dataset"

NUM_CLASSES = 4
NUM_WORKERS = 8

VAL_RATIO = 0.2
SEED = 123
TOTAL_ITER = 15000

# MODEL_YAML = "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
# MODEL_YAML = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
MODEL_YAML = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"