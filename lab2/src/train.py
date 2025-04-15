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
from torch.amp import autocast, GradScaler

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
    
    # return model
    
    # weights = torchvision.models.ResNet50_Weights.DEFAULT
    # backbone = resnet_fpn_backbone(
    #     backbone_name='resnet50',
    #     weights=weights,
    #     trainable_layers=3  # 训练最后3个残差块
    # )
    
    # # Anchor 配置（需匹配 FPN 输出层数）
    # anchor_sizes = ((32,), (64,), (128,), (256,), (512,))  # 对应 FPN 的5个输出层
    # aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    
    # anchor_generator = AnchorGenerator(
    #     sizes=anchor_sizes,
    #     aspect_ratios=aspect_ratios
    # )

    # # 构建模型
    # model = FasterRCNN(
    #     backbone,
    #     num_classes=num_classes,
    #     rpn_anchor_generator=anchor_generator,
    #     box_score_thresh=0.8,
    #     # 关键参数：FPN 输出的通道数（默认256）
    #     box_head_detections_per_img=200  
    # )
    # return model


    weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
    backbone = torchvision.models.mobilenet_v3_large(weights=weights).features
    backbone.out_channels = 960

    # 修改anchor生成器
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    # 構建Faster R-CNN模型
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_score_thresh=0.8  # 提高檢測閾值
    )
    
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
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model().to(device)
    
    import torch.optim as optim
    import torch.optim.lr_scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    epochs = 50
    
    # mobile opti scheduler

    # # 優化器改進
    # optimizer = optim.AdamW(
    #     model.parameters(),
    #     lr=1e-3,                # 初始學習率保持不變
    #     # weight_decay=0.0001,     # 降低權重衰減係數(原0.0005→0.0001)
    #     # betas=(0.9, 0.999),      # 保持默認動量參數
    #     eps=1e-08
    # )

    def reference_optim_scheduler(model):
        # learning rate parameters
        optim_lr = 5e-3
        optim_momentum = 0.9
        optim_weight_decay = 0.0005

        # learning rate schedule
        lr_gamma = 0.33
        lr_dec_step_size = 100

        optimizer = optim.SGD(
            model.parameters(),
            lr=optim_lr,
            momentum=optim_momentum,
            weight_decay=optim_weight_decay
        )

        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=lr_dec_step_size,
            gamma=lr_gamma
        )

        return optimizer, lr_scheduler
    
    optimizer, lr_scheduler = reference_optim_scheduler(model)

    # # 學習率調度器升級
    # lr_scheduler = CosineAnnealingLR(
    #     optimizer,
    #     T_max=epochs*0.6,        # 週期設為總epoch數的60%
    #     eta_min=1e-6             # 最小學習率下限
    # )



    # 添加梯度裁剪(在訓練循環中)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # optim_config = {
    #     'lr': 8e-5,
    #     'betas': (0.9, 0.999),
    #     'weight_decay':  0.2
    # }

    # # 初始化優化器
    # optimizer = optim.AdamW(model.parameters(), **optim_config)

    # total_epochs = epochs
    # num_steps_per_epoch = len(train_data_loader)  # 假設每個epoch有1000個step
    # total_steps = total_epochs * num_steps_per_epoch
    # warmup_steps = int(0.25 * total_epochs * num_steps_per_epoch)  # 前0.25 epochs預熱

    # def lr_lambda(current_step):
    #     # 線性預熱階段
    #     if current_step < warmup_steps:
    #         return current_step / warmup_steps
    #     # 半週期餘弦衰減階段
    #     progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
    #     return 0.5 * (1 + math.cos(math.pi * progress))  # 半週期公式

    # lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    
    
    # scaler = GradScaler(enabled=True)

    train_loss_list = []
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        
        train_loss_iter = []
        bar = tqdm(train_data_loader, desc="Training", leave=False)
        for images, targets in train_data_loader:
            images = [img.to(device) for img in images]  # 列表推導式逐個轉移
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]  # 雙層推導式
            
            with autocast(device):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            train_loss_iter.append(losses.detach().cpu().item())
            
            optimizer.zero_grad()
            # scaler.scale(losses).backward()
            optimizer.step()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # scaler.step(optimizer)

            
            # scaler.update()

            # lr_scheduler.step()

            bar.set_postfix(loss=losses.detach().cpu().item() / len(train_data_loader))
            bar.update()
            
        train_loss_list.append(np.mean(train_loss_iter))
        bar.close()
        
        torch.save(model.state_dict(), f"ckpt/model_epoch_{epoch}.pth")
        
    plot_img(
        train_loss_list,
        'Train Loss',
        'Training Loss',
        'Loss',
        f'img/train_loss.png'
    )

            
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
    import torch.backends
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

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
        'batch_size': 16,
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