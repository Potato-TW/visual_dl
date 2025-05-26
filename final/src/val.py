import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# Constants
ROOT_PATH = 'dataset'
IMG_SIZE = 1024  # Model input size

import cv2  # 新增导入

# def load_test_image(image_id):
#     """加载测试图片并返回原始和预处理版本"""
#     img_path = os.path.join(ROOT_PATH, 'test', f'{image_id}.jpg')
    
#     # 读取原始图片（用于可视化）
#     orig_img = cv2.imread(img_path)
#     original_size = orig_img.shape[:2][::-1]  # (width, height)
    
#     # 预处理图片（用于模型预测）
#     img = cv2.resize(orig_img, (IMG_SIZE, IMG_SIZE))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色通道
    
#     return img, orig_img, original_size  # 返回原始和预处理图片

def draw_bbox(image, boxes, original_size):
    """在原始图片上绘制预测框"""
    # 颜色配置
    BOX_COLOR = (0, 255, 0)  # 绿色边框
    TEXT_COLOR = (0, 0, 255)  # 红色文字
    FONT_SCALE = 0.8
    THICKNESS = 2
    
    img_h, img_w = image.shape[:2]
    orig_w, orig_h = original_size
    
    # 计算缩放比例
    scale_x = orig_w / img_w
    scale_y = orig_h / img_h
    
    for box in boxes:
        # 解析预测结果
        conf = box.conf.item()
        x_center, y_center, width, height = box.xywh[0][:4].tolist()
        
        # 转换到原始尺寸坐标
        x = int((x_center - width/2) * orig_w)
        y = int((y_center - height/2) * orig_h)
        w = int(width * orig_w)
        h = int(height * orig_h)
        
        # 绘制矩形
        cv2.rectangle(image, 
                     (x, y), 
                     (x + w, y + h),
                     BOX_COLOR, 
                     THICKNESS)
        
        # 添加置信度标签
        label = f"{conf:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS)
        
        cv2.rectangle(image, 
                     (x, y - text_height - 10),
                     (x + text_width, y),
                     BOX_COLOR, 
                     -1)  # 填充背景
        
        cv2.putText(image, label,
                   (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   FONT_SCALE,
                   TEXT_COLOR,
                   THICKNESS)
    
    return image

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
        x_center, y_center, width, height = pred.xywhn[0][:4].tolist()
        
        # 转换为原始图像像素坐标（左上角格式）
        x = max(0, (x_center - width/2) * original_width)
        y = max(0, (y_center - height/2) * original_height)
        w = width * original_width
        h = height * original_height
        
        # 格式化为整数坐标
        boxes.append(f"{conf:.1f} {x} {y} {w} {h}")
    
    return f"{image_id},{' '.join(boxes)}"

# model = YOLO("yolo12n.pt")
model_path = "yolo_train_results/exp1n6_bigdata/weights/best.pt"  # 模型路径
confidence_threshold = 0.3  # 置信度阈值

def predict_test_dataset(model_path, confidence_threshold=0.3):
    """Generate predictions for test dataset"""
    # 创建输出目录
    # vis_dir = os.path.join('test_with_bbox')
    # os.makedirs(vis_dir, exist_ok=True)
    
    # 加载模型
    model = YOLO(model_path)
    
    # 获取测试图像ID列表
    test_images = [f.stem for f in Path(os.path.join(ROOT_PATH, 'test')).glob('*.jpg')]
    
    # 生成预测结果
    submission_rows = []
    for image_id in tqdm(test_images, desc='Predicting'):
        # 加载和预处理图像
        img, original_size = load_test_image(image_id)
        # img, orig_img, original_size = load_test_image(image_id)
        
        # 预测
        results = model.predict(
            source=img,
            conf=confidence_threshold,
            imgsz=IMG_SIZE,  # 使用相同的输入尺寸
            verbose=False,
            save=True,
            name=f'{image_id}_pred',
            nms=True,
            agnostic_nms=True,
        )

        print(results[0])
        
        # if len(results[0].boxes) > 0:
        #     vis_img = draw_bbox(orig_img.copy(), results[0].boxes, original_size)
        # else:
        #     vis_img = orig_img  # 无预测框时保存原图
            
        # output_path = os.path.join(vis_dir, f'{image_id}_pred.jpg')
        # cv2.imwrite(output_path, vis_img)

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