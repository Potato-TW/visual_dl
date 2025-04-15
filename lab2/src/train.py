import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torchvision.transforms as T

from dataloader import DigitDataset

from tqdm import tqdm

from matplotlib import pyplot as plt
import numpy as np

def build_model(num_classes=11):
    # # 加載預訓練backbone
    # weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
    # backbone = torchvision.models.mobilenet_v2(weights=weights).features
    # backbone.out_channels = 1280

    # # 修改anchor生成器
    # anchor_generator = AnchorGenerator(
    #     sizes=((32, 64, 128, 256, 512),),
    #     aspect_ratios=((0.5, 1.0, 2.0),)
    # )

    # # 構建Faster R-CNN模型
    # model = FasterRCNN(
    #     backbone,
    #     num_classes=num_classes,
    #     rpn_anchor_generator=anchor_generator,
    #     box_score_thresh=0.8  # 提高檢測閾值
    # )
    
    # # model.load_state_dict(torch.load('ckpt/model_epoch_49.pth', weights_only=True))
    
    # return model
    
    # 使用预训练的ResNet50+FPN backbone
    # backbone = resnet_fpn_backbone(
    #     backbone_name='resnet50',
    #     weights=torchvision.models.ResNet50_Weights.DEFAULT,
    #     trainable_layers=3  # 只微调最后3個block
    # )
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        # num_classes=num_classes,
        box_score_thresh=0.8,
        )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)
    # model.box_score_thresh = 0.8

    # backbone = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
    # backbone = nn.Sequential(*list(backbone.children())[:-2])  # 移除最後兩層
    # backbone.out_channels = 2048  # 對應ResNet50特徵維度
    
    # # 使用官方预设的anchor生成器（针对FPN优化）
    # anchor_sizes = ((32,), (64,), (128,), (256,), (512,))  # 各特徵層的基礎尺寸
    # aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    
    # return FasterRCNN(
    #     backbone,
    #     num_classes=num_classes,
    #     rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
    #     box_score_thresh=0.8,
    #     box_batch_size_per_image=128  # 增加ROI采样数量以提升精度
    # )

    return model

def plot_img(data, data_label, title, y_label, save_path, y_lim=None):
    from matplotlib import pyplot as plt
    plt.plot(data, label=data_label)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel(y_label)
    if y_lim is not None:
        plt.ylim(y_lim[0], y_lim[1])
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    
# 訓練流程
def train(train_data_loader, val_data_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model().to(device)
    

    
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    epochs = 5
    
    # # mobile opti scheduler

    # # 優化器改進
    # optimizer = optim.AdamW(
    #     model.parameters(),
    #     lr=0.001,                # 初始學習率保持不變
    #     weight_decay=0.0001,     # 降低權重衰減係數(原0.0005→0.0001)
    #     betas=(0.9, 0.999),      # 保持默認動量參數
    #     eps=1e-08
    # )

    # # 學習率調度器升級
    # lr_scheduler = CosineAnnealingLR(
    #     optimizer,
    #     T_max=epochs*0.6,        # 週期設為總epoch數的60%
    #     eta_min=1e-6             # 最小學習率下限
    # )

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=5, 
        gamma=0.1
    )

    # 添加梯度裁剪(在訓練循環中)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    best_val_loss = float('inf')
    
    train_loss_list = []
    val_loss_list = []
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        
        train_loss_iter = []
        bar = tqdm(train_data_loader, desc="Training", leave=False)
        for images, targets in train_data_loader:
            images = [img.to(device) for img in images]  # 列表推導式逐個轉移
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]  # 雙層推導式
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            train_loss_iter.append(losses.detach().cpu().item())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            lr_scheduler.step()
            
            bar.set_postfix(loss=losses.detach().cpu().item() / len(train_data_loader))
            bar.update()
            
        train_loss_list.append(np.mean(train_loss_iter))
        bar.close()
        
        # model.eval()
        
        # val_bar = tqdm(val_data_loader, desc="Validation", leave=False)
        # with torch.no_grad():
        #     for images, targets in val_data_loader:
        #         images = [img.to(device) for img in images]
        #         targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
                
        #         val_loss_dict = model(images, targets)
        #         val_loss_dict = model(images)
                
        #         print(val_loss_dict)
                
        #         val_losses = sum(loss for loss in val_loss_dict.values())
        #         val_loss_list.append(val_losses.detach().cpu().item())
                
        #         val_bar.set_postfix(loss=val_losses.detach().cpu().item() / len(val_data_loader))
        #         val_bar.update()
        # val_bar.close()
        
        # # 保存模型
        # if val_losses < best_val_loss:
        #     best_val_loss = val_losses
        #     # print(f"Saving model with loss: {best_val_loss.item()}")
        torch.save(model.state_dict(), f"ckpt/model_epoch_{50+epoch}.pth")
        
    plot_img(
        train_loss_list,
        'Train Loss',
        'Training Loss',
        'Loss',
        f'img/train_loss.png'
    )
        # plot_img(
        #     val_loss_list,
        #     'Val Loss',
        #     'Val Loss',
        #     'Loss',
        #     f'img/val_loss.png'
        # )
            

def custom_collate(batch):
    """處理可變長度目標檢測數據"""
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append({
            'boxes': target['boxes'],
            'labels': target['labels'],
            'image_id': target['image_id']
        })
    
    # 堆疊圖像張量
    images = torch.stack(images, dim=0)
    return images, targets

if __name__ == "__main__":
    import os
    os.makedirs('ckpt/', exist_ok=True)
    os.makedirs('img/', exist_ok=True)
    
    dataset = DigitDataset(
        root='dataset/train',
        annotation_path='dataset/train.json',
        transforms=T.Compose([
            # T.Resize(size=232, antialias=True),
            # T.CenterCrop(size=224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )
    
    print(f'Dataset length: {len(dataset)}')
    # print(f'Sample Image: {dataset[0][1]}')

    # print(dataset.validate_transformation(dataset))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loader_param = {
        'batch_size': 8,
        'num_workers': 12,
        'persistent_workers': True,
        'pin_memory': 'cuda' in device,
        'pin_memory_device': device if 'cuda' in device else '',
        'collate_fn': custom_collate,
    }

    train_data_loader = torch.utils.data.DataLoader(
        dataset,
        **loader_param,
        shuffle=True,
    )
    
    val_data_loader = torch.utils.data.DataLoader(
        dataset,
        **loader_param,
        shuffle=False,
    )

    train(train_data_loader, val_data_loader)