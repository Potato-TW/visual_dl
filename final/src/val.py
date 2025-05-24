import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# Constants
ROOT_PATH = 'dataset/images'
IMG_SIZE = 416  # Model input size

def load_test_image(image_id):
    """Load and preprocess test image"""
    img_path = os.path.join(ROOT_PATH, 'test', f'{image_id}.jpg')
    img = Image.open(img_path)
    original_size = img.size  # 保存原始尺寸
    # 调整到模型输入尺寸
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img), original_size

def convert_yolo_to_submission(predictions, image_id, original_size):
    """Convert YOLO predictions to submission format
    
    Format: image_id,confidence x y width height [confidence x y width height ...]
    Example: ce4833752,0.5 0 0 100 100
    Multiple boxes: 1da9078c1,0.3 0 0 50 50 0.5 10 10 30 30
    Empty prediction: 6ca7b2650,
    """
    if len(predictions) == 0:
        return f"{image_id},"
    
    original_width, original_height = original_size
    boxes = []
    
    for pred in predictions:
        # 获取置信度
        conf = pred.conf.item()
        
        # 获取归一化的边界框坐标（xywh格式）
        x_center, y_center, width, height = pred.xywh[0][:4].tolist()
        
        # 转换为原始图像像素坐标（左上角格式）
        x = max(0, int((x_center - width/2) * original_width))
        y = max(0, int((y_center - height/2) * original_height))
        w = int(width * original_width)
        h = int(height * original_height)
        
        # 格式化为整数坐标
        boxes.append(f"{conf:.1f} {x} {y} {w} {h}")
    
    return f"{image_id},{' '.join(boxes)}"

# model = YOLO("yolo12n.pt")
model_path = "yolo_train_results/exp17/weights/last.pt"  # 模型路径
confidence_threshold = 0.3  # 置信度阈值

def predict_test_dataset(model_path, confidence_threshold=0.3):
    """Generate predictions for test dataset"""
    # 加载模型
    model = YOLO(model_path)
    
    # 获取测试图像ID列表
    test_images = [f.stem for f in Path(os.path.join(ROOT_PATH, 'test')).glob('*.jpg')]
    
    # 生成预测结果
    submission_rows = []
    for image_id in tqdm(test_images, desc='Predicting'):
        # 加载和预处理图像
        img, original_size = load_test_image(image_id)
        
        # 预测
        results = model.predict(
            source=img,
            conf=confidence_threshold,
            imgsz=IMG_SIZE,  # 使用相同的输入尺寸
            verbose=False
        )
        
        # 转换预测结果为提交格式
        submission_row = convert_yolo_to_submission(results[0].boxes, image_id, original_size)
        submission_rows.append(submission_row)
    
    # 创建提交DataFrame
    submission_df = pd.DataFrame(
        [row.split(',') for row in submission_rows], 
        columns=['image_id', 'PredictionString']
    )
    
    return submission_df


submission_df = predict_test_dataset(
    model_path=model_path,
    confidence_threshold=0.3
)

# 保存提交文件
submission_path = './submission.csv'
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")