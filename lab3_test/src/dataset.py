import random
import re
from pathlib import Path
from typing import Dict, List

# import skimage.io as sio
import cv2
import numpy as np
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

from config import *
from utils import encode_mask

def get_item(dir_path: Path, img_id: int) -> Dict:
    img_path = dir_path / "image.tif"
    img = cv2.imread(str(img_path))
    h, w = img.shape[0], img.shape[1]
    record = {
        "file_name": str(img_path),
        "image_id": img_id,
        "height": h,
        "width": w,
    }

    annos = []
    for mask_path in dir_path.glob("class*.tif"):
        m = re.search(r"class(\d+)\.tif", mask_path.name)
        class_ = int(m.group(1)) - 1
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask.ndim > 2:
            mask = mask[..., 0]
        for inst_id in np.unique(mask):
            if inst_id == 0:
                continue
            bin_mask = (mask == inst_id).astype(np.uint8).copy()
            ys, xs = np.where(bin_mask)
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max() # xyxy
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

    record["annotations"] = annos
    return record


def get_dict(suffix: str) -> List[Dict]:
    dirs = list((DATA_ROOT / "train").iterdir())
    random.Random(SEED).shuffle(dirs)
    end = int(len(dirs) * (1 - VAL_RATIO))
    splited_dir = dirs[:end] if suffix == "train" else dirs[end:]
    return [get_item(dir, i) for i, dir in enumerate(splited_dir)]


def register_dataset(name: str, suffix: str):
    if name in DatasetCatalog.list():
        return
        
    DatasetCatalog.register(name, lambda s=suffix: get_dict(s))
    MetadataCatalog.get(name).set(
        thing_classes=[f'class{i}' for i in range(1, NUM_CLASSES+1)],
        mask_format="bitmask",
    )
