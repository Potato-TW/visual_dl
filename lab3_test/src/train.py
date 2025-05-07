import argparse
import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator

from config import *
from dataset import register_dataset


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, 'coco')
        return COCOEvaluator(dataset_name, cfg, output_dir=output_folder)


def setup(args):
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

    # cfg.INPUT.MIN_SIZE_TRAIN=(224,)
    cfg.INPUT.MIN_SIZE_TRAIN=(512,)
    cfg.INPUT.MAX_SIZE_TRAIN=1024
    # cfg.INPUT.MAX_SIZE_TRAIN=512
    cfg.INPUT.MIN_SIZE_TEST=0
    cfg.INPUT.MAX_SIZE_TEST=1024
    # cfg.INPUT.MAX_SIZE_TEST=512

    cfg.TEST.EVAL_PERIOD=1000

    cfg.OUTPUT_DIR=str(OUTPUT_DIR.resolve())
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.freeze()
    default_setup(cfg, args)

    return cfg


def main(args):
    register_dataset("dataset_train", "train")
    register_dataset("dataset_val", "val")

    cfg=setup(args)
    trainer=Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if args.eval_only:
        trainer.test(cfg,trainer.model)
    else:
        trainer.train()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--num-machine', type=int, default=1)
    parser.add_argument('--machine-rank', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--dist-url', default='auto')
    args = parser.parse_args()

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machine,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,)
        )
