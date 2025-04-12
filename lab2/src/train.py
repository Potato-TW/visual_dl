import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import torchvision.transforms as T

from dataloader import DigitDataset

from tqdm import tqdm

def build_model(num_classes=11):
    # 加載預訓練backbone
    weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
    backbone = torchvision.models.mobilenet_v2(weights=weights).features
    backbone.out_channels = 1280

    # 修改anchor生成器
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # 構建Faster R-CNN模型
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_score_thresh=0.8  # 提高檢測閾值
    )
    
    return model


# 訓練流程
def train(data_loader):
    model = build_model()
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    epochs = 10
    
    for epoch in tqdm(range(epochs), desc="Epochs"):
        bar = tqdm(data_loader, desc="Training", leave=False)
        for images, targets in data_loader:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            bar.set_postfix(loss=losses.item())
            bar.update()
            
        bar.close()

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
        # 'pin_memory': 'cuda' in device,
        # 'pin_memory_device': device if 'cuda' in device else '',
        'collate_fn': custom_collate,
    }

    data_loader = torch.utils.data.DataLoader(
        dataset,
        **loader_param,
        shuffle=True,
    )
    
    train(data_loader)