import torch

from torchvision.ops import box_iou
from collections import defaultdict
import numpy as np

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torchvision.transforms as T

from dataloader import DigitDataset

from PIL import Image

import json

import torchvision.transforms as T

import os

from tqdm import tqdm

from matplotlib import pyplot as plt
import numpy as np
from torch.amp import autocast, GradScaler


def evaluate(model, data_loader, device, iou_threshold=0.5):
    """目标检测评估函数"""
    model.eval()
    results = defaultdict(list)
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            # 数据迁移
            images = [img.to(device) for img in images]
            
            # 模型预测
            with autocast(device_type='cuda' if 'cuda' in device else 'cpu'):
                predictions = model(images)
            
            # 逐样本处理
            for pred, true in zip(predictions, targets):
                # 迁移标签到设备
                true_boxes = true['boxes'].to(device)
                true_labels = true['labels'].to(device)
                
                # 计算IOU矩阵
                iou_matrix = box_iou(pred['boxes'], true_boxes)
                
                # 匹配检测与真实框
                matched_true = set()
                for det_idx in range(len(pred['boxes'])):
                    if len(iou_matrix[det_idx]) == 0:
                        continue
                    
                    best_true_idx = iou_matrix[det_idx].argmax()
                    if iou_matrix[det_idx][best_true_idx] >= iou_threshold:
                        if best_true_idx not in matched_true:
                            # 记录匹配结果
                            results['pred_labels'].append(pred['labels'][det_idx].item())
                            results['true_labels'].append(true_labels[best_true_idx].item())
                            matched_true.add(best_true_idx)
                
                # 统计未匹配的真实框（假阴性）
                for true_idx in range(len(true_boxes)):
                    if true_idx not in matched_true:
                        results['pred_labels'].append(-1)  # 标记为未检测
                        results['true_labels'].append(true_labels[true_idx].item())
    
    # 计算指标
    pred_labels = np.array(results['pred_labels'])
    true_labels = np.array(results['true_labels'])
    
    # 准确率计算（仅考虑匹配的检测）
    valid_mask = pred_labels != -1
    accuracy = np.mean(pred_labels[valid_mask] == true_labels[valid_mask]) if any(valid_mask) else 0
    
    # 召回率计算
    recall = len(matched_true) / len(true_labels) if len(true_labels) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'recall': recall,
        'matched_samples': len(matched_true),
        'total_samples': len(true_labels)
    }