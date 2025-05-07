import logging
import argparse
import os
from tabulate import tabulate
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from config import *
from dataset import *
from detectron2.engine import DefaultTrainer, default_setup, launch

def print_model_params(cfg):
    """ 分層次打印模型參數規模 """
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    
    # 參數統計邏輯
    param_groups = {}
    for name, param in model.named_parameters():
        module_name = name.split('.')[0]  # 提取頂層模塊名
        num_params = param.numel()
        param_groups.setdefault(module_name, 0)
        param_groups[module_name] += num_params
    
    # 可視化表格
    table_data = []
    total_params = 0
    for module, params in param_groups.items():
        table_data.append([module, f"{params:,}", f"{params/1e6:.2f}M"])
        total_params += params
    
    # 打印結果
    logger = logging.getLogger(__name__)
    logger.info("\n" + tabulate(table_data, 
               headers=["Module", "Parameters", "Scale"],
               tablefmt="fancy_grid"))
    logger.info(f"Total Trainable Params: {total_params/1e6:.2f}M")

# 在訓練腳本中調用
from detectron2.engine import DefaultTrainer

class MyTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        print_model_params(cfg)  # 初始化時打印參數

def setup():
    cfg=get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_YAML))
    cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url(MODEL_YAML)
    cfg.INPUT.MASK_FORMAT="bitmask"

    cfg.DATASETS.TRAIN=(f'{DATASET_NAME}_train',)
    cfg.DATASETS.TEST=(f'{DATASET_NAME}_val',)
    cfg.DATALOADER.NUM_WORKERS=NUM_WORKERS

    cfg.MODEL.ROI_HEADS.NUM_CLASSES=NUM_CLASSES

    cfg.SOLVER.IMS_PER_BATCH=1
    cfg.SOLVER.BASE_LR=4e-4
    cfg.SOLVER.MAX_ITER=TOTAL_ITER
    cfg.SOLVER.STEPS=(5000,)
    cfg.SOLVER.CHECKPOINT_PERIOD=1000

    cfg.INPUT.MIN_SIZE_TRAIN=(512,)
    cfg.INPUT.MAX_SIZE_TRAIN=1024
    cfg.INPUT.MIN_SIZE_TEST=0
    cfg.INPUT.MAX_SIZE_TEST=1024

    cfg.TEST.EVAL_PERIOD=1000

    # cfg.OUTPUT_DIR=str(OUTPUT_DIR.resolve())
    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.freeze()
    default_setup(cfg, args)

    return cfg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--num-machine', type=int, default=1)
    parser.add_argument('--machine-rank', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--dist-url', default='auto')
    args = parser.parse_args()

    register_dataset("dataset_train", "train")
    register_dataset("dataset_val", "val")

    cfg = setup()
    MyTrainer(cfg)


