# train.py
from ultralytics import YOLO, settings
import pandas as pd
import os
import shutil

settings.update({"wandb": True})

dataset_dir = '/home/bhg/visual_dl/final/dataset'
# 創建 YOLO 數據配置文件
data_yaml = os.path.join(dataset_dir, 'data.yaml')
with open(data_yaml, 'w') as f:
    f.write(f'''\
train: {'/home/bhg/visual_dl/final/dataset/images/train'}
val: {'/home/bhg/visual_dl/final/dataset/images/val'}  # 需要自行創建驗證集

nc: 1  # 類別數量
names: ['object']  # 類別名稱
''')

# 初始化模型
model = YOLO('yolo12n.pt')  # 使用 nano 版本模型

# 訓練配置
results = model.train(
    data=data_yaml,
    epochs=400,
    imgsz=640,
    batch=32,
    device='0',  # 使用 GPU
    project='yolo_train_results',
    name='exp1'
)