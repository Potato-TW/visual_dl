"""
Detectron2 dataset registration for cell instance segmentation.
"""

import random
import re
from pathlib import Path
from typing import Dict, List

# import skimage.io as sio
import cv2
import numpy as np
from detectron2.structures import BoxMode

from config import *

from utils import encode_mask


# def _annos_from_mask(mask_path: Path, cat_id: int) -> List[Dict]:
#     from pycocotools import mask as mask_utils

#     # img = sio.imread(str(mask_path))
#     mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
#     if mask.ndim > 2:
#         mask = mask[..., 0]
#     annos = []
#     for inst_id in np.unique(mask):
#         if inst_id == 0:
#             continue
#         # bin_mask = (mask == inst_id).astype(np.uint8)
#         bin_mask = (mask == inst_id).astype(np.uint8).copy()
#         ys, xs = np.where(bin_mask)
#         x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max() # xyxy
#         # rle = mask_utils.encode(np.asfortranarray(bin_mask))
#         # rle["counts"] = rle["counts"].decode("utf-8")
#         rle = encode_mask(bin_mask)
#         annos.append(
#             {
#                 "bbox": [int(x0), int(y0), int(x1), int(y1)],
#                 "bbox_mode": BoxMode.XYXY_ABS,
#                 "segmentation": rle,
#                 "category_id": cat_id,
#                 "iscrowd": 0,
#             }
#         )
#     return annos


def get_item(dir_path: Path, img_id: int) -> Dict:
    import cv2

    img_path = dir_path / "image.tif"
    img = cv2.imread(str(img_path))
    h, w = img.shape[0], img.shape[1]
    record = {
        "file_name": str(img_path),
        "image_id": img_id,
        "height": h,
        "width": w,
        # "annotations": [],
    }

    annos = []
    for mask_path in dir_path.glob("class*.tif"):
        m = re.search(r"class(\d+)\.tif", mask_path.name)
        # if not m:
        #     continue
        class_ = int(m.group(1)) - 1
        # if class_ > NUM_CLASSES:
        #     continue
        # record["annotations"] += _annos_from_mask(mask_path, class_ - 1)
        # tmp += _annos_from_mask(mask_path, class_ - 1)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask.ndim > 2:
            mask = mask[..., 0]
        # tmp = []
        for inst_id in np.unique(mask):
            if inst_id == 0:
                continue
            # bin_mask = (mask == inst_id).astype(np.uint8)
            bin_mask = (mask == inst_id).astype(np.uint8).copy()
            ys, xs = np.where(bin_mask)
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max() # xyxy
            # rle = mask_utils.encode(np.asfortranarray(bin_mask))
            # rle["counts"] = rle["counts"].decode("utf-8")
            rle = encode_mask(bin_mask)
            annos.append(
                {
                    "bbox": [int(x0), int(y0), int(x1), int(y1)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": rle,
                    "category_id": class_,
                    "iscrowd": 0,
                }
            )
        # annos += tmp

    record["annotations"] = annos
    return record


def get_dict(suffix: str) -> List[Dict]:
    dirs = list((DATA_ROOT / "train").iterdir())
    random.Random(SEED).shuffle(dirs)
    # if suffix == "trainval":
    #     chosen = dirs
    # else:
    end = int(len(dirs) * (1 - VAL_RATIO))
    splited_dir = dirs[:end] if suffix == "train" else dirs[end:]
    return [get_item(dir, i) for i, dir in enumerate(splited_dir)]


def register_dataset(name: str, suffix: str):
    from detectron2.data import DatasetCatalog, MetadataCatalog

    # if name in DatasetCatalog.list():
    #     return
        
    DatasetCatalog.register(name, lambda s=suffix: get_dict(s))
    MetadataCatalog.get(name).set(
        # thing_classes=[CLASS_NAME_MAP[i + 1] for i in range(NUM_CLASSES)],
        thing_classes=[f'class{i}' for i in range(1, NUM_CLASSES+1)],
        mask_format="bitmask",
    )


def register_cell_dataset():
    register_dataset("dataset_train", "train")
    register_dataset("dataset_val", "val")
    # register_dataset("dataset_trainval", "trainval")
