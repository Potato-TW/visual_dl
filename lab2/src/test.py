import torch

import torchvision
# from torchvision.models.detection import fasterrcnn_resnet50_fpn
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torch.utils.data import Dataset
from dataloader import DigitDataset
from train import custom_collate

from PIL import Image

import json
from tqdm import tqdm

import torchvision.transforms as T

import os

class TestDataset(Dataset):
    def __init__(self, root, annotation_path=None, transforms=None):
        self.root = root
        self.transforms = transforms
        
        self.image_info = self._scan_directory()
    
    def _scan_directory(self):
        import re 
        
        image_files = [f for f in os.listdir(self.root) 
                      if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        
        image_info = {}
        for f in sorted(image_files):
            match = re.search(r'^(\d+)', f)
            if match:
                img_id = int(match.group(1))
                image_info[img_id] = {'id': img_id, 'file_name': f}

        
        return image_info

    def __getitem__(self, idx):
        image_id = list(self.image_info.keys())[idx]
        img_path = f"{self.root}/{self.image_info[image_id]['file_name']}"
        
        image = Image.open(img_path).convert('RGB')
        
        transform_chain = T.Compose([
            T.Resize((224, 224)),      
            T.ToTensor(),              
            T.Normalize(               
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image_tensor = transform_chain(image)
        
        return image_tensor, {'image_id': torch.tensor([image_id])}

    def __len__(self):
        return len(self.image_info)
    
    def get_info(self):
        return self.image_info


def load_model(ckpt_path, num_classes=11):
    weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
    backbone = torchvision.models.mobilenet_v2(weights=weights).features
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_score_thresh=0.8
    )
    
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    
    return model

def load_model_mobilenet_v3(ckpt_path, num_classes=11):
    weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
    backbone = torchvision.models.mobilenet_v3_large(weights=weights).features
    backbone.out_channels = 960

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_score_thresh=0.8
    )

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    return model

def load_model_resnet50(ckpt_path, num_classes=11):
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    backbone = resnet_fpn_backbone(
        backbone_name='resnet50',
        weights=weights,
        trainable_layers=3
    )
    
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_score_thresh=0.8,
        box_head_detections_per_img=200  
    )

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    return model


def deNormalize_box(image_id, xyxy_box):
    new_w, new_h = 224, 224
    
    from PIL import Image
    orig_img = Image.open(f'./dataset/test/{int(image_id.item())}.png')
    orig_w, orig_h = orig_img.size
    
    from torchvision.ops import box_convert
    xywh_box = box_convert(xyxy_box, in_fmt='xyxy', out_fmt='xywh')
    
    xywh_box = xywh_box.tolist()
    
    scale_w = orig_w / new_w
    scale_h = orig_h / new_h
    
    x, y, w, h = xywh_box
    orig_x = x * scale_w
    orig_y = y * scale_h
    orig_w = w * scale_w
    orig_h = h * scale_h  
    
    orig_box = [orig_x, orig_y, orig_w, orig_h]
    
    # print(f'orig box: {orig_box}')
    
    return orig_box
    

def convert_to_coco_format(outputs, image_ids, score_threshold=0.8):
    coco_results = []
    
    for img_id, detections in zip(image_ids, outputs):
        boxes = detections['boxes'].cpu().detach()#.numpy()  # (N,4) tensor -> numpy
        scores = detections['scores'].cpu().detach().numpy().astype(float)  # (N,)
        labels = detections['labels'].cpu().detach().numpy().astype(int)  # (N,)
        
        if boxes.shape[0] == 0:
            print(f'img_id: {img_id} no boxes')
            # continue
        
        for i in range(boxes.shape[0]):
            # if scores[i] < score_threshold:
            #     continue
                
            # from torchvision.ops import box_convert
            # new_box = box_convert(boxes[i], in_fmt='xyxy', out_fmt='xywh')
            new_box = deNormalize_box(img_id, boxes[i])
                
            # (x1,y1,x2,y2) -> (x,y,width,height)
            # x1, y1, x2, y2 = boxes[i].tolist()
            # w = x2 - x1
            # h = y2 - y1
            
            output_dict = {
                "image_id": int(img_id.item()), 
                "bbox": new_box,
                "score": float(scores[i]),
                "category_id": int(labels[i]),
            }
            
            coco_results.append(output_dict)
    
    return coco_results

def task2_do(outputs, image_ids, task2, score_threshold=0.8):
    import numpy as np
    # print(f'coco: {coco}')
    
    for img_id, detections in zip(image_ids, outputs):
        img_id = img_id.item()
        # print(f'img_id: {type(img_id), img_id}')
        boxes = detections['boxes'].cpu().detach().numpy()  # (N,4) tensor -> numpy
        # print(f'boxes: {boxes, boxes.shape}')
        scores = detections['scores'].cpu().detach().numpy().astype(float)  # (N,)

        
        if boxes.shape[0] == 0:
            continue
        
        labels_tmp = detections['labels'].cpu().detach().numpy().astype(int)  # (N,)    
        labels_tmp -= 1
        
        # print(f'labels_tmp: {labels_tmp}')
        
        new_boxes = []
        labels = []
        for i in range(boxes.shape[0]):
            # if scores[i] < score_threshold:
            #     continue
            new_boxes.append(boxes[i][0])
            labels.append(labels_tmp[i])
        
        # scores = detections['scores'].cpu().detach().numpy().astype(float)  # (N,)

        # print(f'new_boxes: {new_boxes}')
        # # print(f'labels: {labels, labels.shape}')
        
        # print(f'labels: {labels}')
        
        new_boxes = np.array(new_boxes)
        labels = np.array(labels)
        
        # if new_boxes.shape[0] == 0:
        #     continue
        
        sort_idx = np.argsort(new_boxes)

        # a_sorted = boxes[sort_idx]
        b_sorted = labels[sort_idx]
        
        # print(a_sorted)
        # print(b_sorted)
            
        pred = int(''.join(b_sorted.astype(str)))
        # print(f'pred: {pred}')
        
        # if not isinstance(img_id, int):
        #     print(f'not int: {img_id}')
        # if not isinstance(pred, int):
        #     print(f'not int: pred {pred}')
        # print(f'pred: {pred}')
        task2.loc[task2['image_id'] == img_id, 'pred_label'] = pred
        
    return task2

def test(model, test_data_loader):
    model.eval()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    task1 = []
    

    content = {
        "image_id": range(1, 13069),
        "pred_label": -1      
    }

    import pandas as pd

    task2 = pd.DataFrame(content)
    
    test_bar = tqdm(test_data_loader, desc="Test", leave=False)
    with torch.no_grad():
        for images, images_id in test_data_loader:
            images = images.to(device)

            tmp = model(images)
            # print(tmp)
            
            coco = convert_to_coco_format(tmp, images_id['image_id'], score_threshold=0.8)
            
            task1 = task1 + coco
            
            task2 = task2_do(tmp, images_id['image_id'], task2, score_threshold=0.8)
            
            test_bar.update()
    test_bar.close()
    
    with open('./result/task1/pred.json', 'w') as f:
        json.dump(task1, f, indent=4)
        
    task2.to_csv('./result/task2/pred.csv', index=False)

        
if __name__ == "__main__":
    import os
    os.makedirs('result/task1/', exist_ok=True)
    os.makedirs('result/task2/', exist_ok=True)
    with open ('./result/task1/pred.json', 'w') as f:
        pass
    
    test_dataset = TestDataset(
        root='dataset/test',
    )
    
    # print(test_dataset.get_info())
    
    print(f'Dataset length: {len(test_dataset)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loader_param = {
        'batch_size': 8,
        'num_workers': 12,
        'persistent_workers': True,
        'pin_memory': 'cuda' in device,
        'pin_memory_device': device if 'cuda' in device else '',
        # 'collate_fn': custom_collate,
    }

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        **loader_param,
        shuffle=False,
    )

    ckpt_path = '/home/bhg/visual_dl/lab2/ckpt/model_epoch_0.pth'
    model = load_model(ckpt_path, 11).to(device)
    # ckpt_path = '/home/bhg/visual_dl/lab2/ckpt/model_epoch_2.pth'
    # model = load_model_mobilenet_v3(ckpt_path, 11).to(device)
    # ckpt_path= '/home/bhg/visual_dl/lab2/ckpt/best_model.pth'
    # model = load_model_resnet50(ckpt_path, 11).to(device)

    
    # print(model)
       
    test(model, test_data_loader)