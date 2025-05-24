# train.py
from ultralytics import YOLO, settings
import pandas as pd
import os
import shutil

settings.update({"wandb": True})

def convert_csv_to_yolo(csv_path, images_dir, labels_dir):
    # 創建標籤目錄
    os.makedirs(labels_dir, exist_ok=True)
    
    # 讀取 CSV 文件
    df = pd.read_csv(csv_path)
    
    # 處理每張圖片
    for (image_id, group) in df.groupby('image_id'):
        # 圖片路徑
        img_path = os.path.join(images_dir, f"{image_id}.jpg")  # 假設圖片格式為 jpg
        
        # 跳過不存在的圖片
        if not os.path.exists(img_path):
            continue
            
        # 創建標籤文件
        txt_path = os.path.join(labels_dir, f"{image_id}.txt")
        
        with open(txt_path, 'w') as f:
            for _, row in group.iterrows():
                # 解析原始坐標 (假設 CSV 中的 bbox 格式為 [x_min, y_min, width, height])
                x_min, y_min, width, height = eval(row['bbox'])
                
                # 計算歸一化坐標
                img_width = row['width']
                img_height = row['height']
                
                x_center = (x_min + width / 2) / img_width
                y_center = (y_min + height / 2) / img_height
                norm_width = width / img_width
                norm_height = height / img_height
                
                # 寫入標籤文件 (假設只有一個類別，class_id=0)
                f.write(f"0 {x_center} {y_center} {norm_width} {norm_height}\n")

# 數據集配置
dataset_dir = 'dataset'
csv_path = 'dataset/train.csv'  # 替換為您的 CSV 文件路徑

# 轉換訓練集
convert_csv_to_yolo(
    csv_path=csv_path,
    images_dir=os.path.join(dataset_dir, 'train'),
    labels_dir=os.path.join(dataset_dir, 'train', 'labels')
)


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