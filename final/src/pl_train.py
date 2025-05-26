# train.py
import argparse
from ultralytics import YOLO, settings
import pandas as pd
import os
import shutil

settings.update({"wandb": False})


hyper_params = {
    # 優化器參數
    'lr0': 0.0032,
    'lrf': 0.12,
    'momentum': 0.843,
    'weight_decay': 0.00036,
    
    # 熱身階段配置
    'warmup_epochs': 2.0,
    'warmup_momentum': 0.5,
    'warmup_bias_lr': 0.05,
    
    # 損失函數權重
    'box': 0.0296,
    'cls': 0.243,

    # 數據增強參數
    'hsv_h': 0.0138,
    'hsv_s': 0.664,
    'hsv_v': 0.464,
    'degrees': 0.373,
    'translate': 0.245,
    'scale': 0.898,
    'shear': 0.602,
    'perspective': 0.0,
    'flipud': 0.5,
    'fliplr': 0.5,
    
    # 進階增強策略
    'mosaic': 1.0,
    'mixup': 0.3
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_yaml', type=str, default=None,
                        help='數據配置文件路徑')
    parser.add_argument('--ckpt', type=str, default='yolo12n.pt')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--name', type=str, default='exp1n')
    args = parser.parse_args()

    print(f"使用的模型: {args.ckpt}")
    model = YOLO(args.ckpt)
    # 訓練配置
    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=1024,
        batch=args.batch_size,
        device='0',  # 使用 GPU
        project='yolo_train_results',
        name=args.name,
        **hyper_params,
    )