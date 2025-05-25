# train.py
from ultralytics import YOLO, settings
import pandas as pd
import os
import shutil

settings.update({"wandb": True})

# dataset_dir = './dataset'
# # 創建 YOLO 數據配置文件
# data_yaml = os.path.join(dataset_dir, 'data.yaml')
# with open(data_yaml, 'w') as f:
#     f.write(f'''\
# train: {'images/train'}
# val: {'images/val'}  # 需要自行創建驗證集

# nc: 1  # 類別數量
# names: ['object']  # 類別名稱
# ''')

data_yaml = 'wheat2020.yaml'  # 假設已經存在 data.yaml 文件
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

# 初始化模型
model = YOLO('yolo12n.pt')  # 使用 nano 版本模型

# 訓練配置
results = model.train(
    data=data_yaml,
    epochs=400,
    imgsz=1024,
    batch=8,
    device='0',  # 使用 GPU
    project='yolo_train_results',
    name='exp1n',
    **hyper_params,
)