import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# train.py
from ultralytics import YOLO, settings
import pandas as pd
import os
import shutil

def generate_pseudo_labels(model, unlabeled_dir, conf_thresh=0.7, iou_thresh=0.5):
    """
    生成偽標籤的核心函數
    :param model: 訓練好的YOLO模型
    :param unlabeled_dir: 未標註圖片目錄 (需包含images子目錄)
    :param conf_thresh: 置信度閾值 (0-1)
    :param iou_thresh: NMS的IoU閾值 (0-1)
    :return: 生成的偽標籤數量統計
    """
    # 初始化路徑
    img_dir = Path(unlabeled_dir) / 'images'
    label_dir = Path(unlabeled_dir) / 'labels'
    label_dir.mkdir(parents=True, exist_ok=True)
    
    # 獲取圖片列表
    img_paths = list(img_dir.glob('*.*'))
    if not img_paths:
        raise ValueError(f"No images found in {img_dir}")
    
    total_boxes = 0
    valid_images = 0
    
    # 處理每張圖片
    for img_path in tqdm(img_paths, desc='Generating pseudo labels'):
        # 推理預測
        results = model.predict(img_path, conf=conf_thresh, iou=iou_thresh, verbose=False)
        
        # 提取預測結果
        boxes = results[0].boxes.xywhn.cpu().numpy()  # 歸一化坐標
        conf = results[0].boxes.conf.cpu().numpy()
        cls = results[0].boxes.cls.cpu().numpy().astype(int)
        
        # 過濾低置信度預測
        keep_idx = np.where(conf >= conf_thresh)[0]
        if len(keep_idx) == 0:
            continue
            
        # 寫入標籤文件
        txt_path = label_dir / f"{img_path.stem}.txt"
        with open(txt_path, 'w') as f:
            for idx in keep_idx:
                x, y, w, h = boxes[idx]
                c = cls[idx]
                f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                total_boxes += 1
        
        valid_images += 1
    
    return {
        'total_images': len(img_paths),
        'labeled_images': valid_images,
        'total_boxes': total_boxes,
        'labeling_rate': valid_images / len(img_paths)
    }

def update_training_data(src_dir, dst_root):
    """
    將偽標籤數據合併到訓練集
    :param src_dir: 偽標籤數據根目錄 (需包含images/labels子目錄)
    :param dst_root: 目標數據集根目錄
    """
    # 初始化路徑
    src_img = Path(src_dir) / 'images'
    src_label = Path(src_dir) / 'labels'
    dst_img_train = Path(dst_root) / 'images/train'
    dst_label_train = Path(dst_root) / 'labels/train'
    
    # 複製圖片和標籤
    for img_path in src_img.glob('*.*'):
        shutil.copy(img_path, dst_img_train / img_path.name)
    for label_path in src_label.glob('*.txt'):
        shutil.copy(label_path, dst_label_train / label_path.name)

# 使用範例 (整合到訓練流程中)
def train_with_pseudo_labeling(train_args):
    # 初始訓練
    model = YOLO('yolo_train_results/exp1n4_0.44private/weights/best.pt')
    # model.train(    
    #     data='dataset/data.yaml',
    #     epochs=10,
    #     imgsz=1024,
    #     batch=8,
    #     device='0',  # 使用 GPU
    #     project='yolo_train_results',
    #     name='exp1n_pseudo_pre'
    #     )
    
    # 迭代偽標籤流程
    for epoch in range(10):  # 可調整迭代次數
        # 生成偽標籤
        pseudo_stats = generate_pseudo_labels(
            model=model,
            unlabeled_dir='dataset/images/unlabeled',  # 未標註數據目錄
            conf_thresh=0.75 - epoch*0.1  # 隨迭代降低閾值
        )
        
        # 合併到訓練集
        update_training_data(
            src_dir='dataset/images/unlabeled',
            dst_root='dataset'
        )
        
        # 更新訓練配置
        with open('dataset/data.yaml', 'a') as f:
            f.write(f"\n# Pseudo Labeling Round {epoch+1}\n")
            f.write(f"pseudo_images: {pseudo_stats['labeled_images']}\n")
            f.write(f"pseudo_boxes: {pseudo_stats['total_boxes']}\n")
        
        # 重新訓練
        model = YOLO('yolo_train_results/exp1n4_0.44private/weights/best.pt')
        model.train(**train_args)

if __name__ == "__main__":
    # 設定訓練參數
    train_args = {
        'data': 'dataset/data.yaml',
        'epochs': 30,
        'imgsz': 1024,
        'batch': 8,
        'device': '0',  # 使用GPU
        'project': 'yolo_train_results',
        'name': 'exp1n_pseudo',
    }
    
    # 執行訓練和偽標籤生成
    train_with_pseudo_labeling(train_args)