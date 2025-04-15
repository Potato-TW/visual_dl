import json
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset

import torchvision.transforms as T
from torchvision.transforms import functional as F

        
class DigitDataset(Dataset):
    def __init__(self, root, annotation_path, transforms=None):
        """
        Args:
            root (str): 圖片根目錄
            annotation_path (str): COCO格式標注文件路徑
            transforms (callable): 數據增強處理
        """
        with open(annotation_path) as f:
            self.coco_data = json.load(f)
        self.root = root
        self.transforms = transforms
        self.image_info = {i['id']: i for i in self.coco_data['images']}
        self.annotations = self._load_annotations()

    def _load_annotations(self):
        ann_dict = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in ann_dict:
                ann_dict[image_id] = []
            ann_dict[image_id].append({
                'bbox': ann['bbox'],
                'category_id': ann['category_id'],
            })
        return ann_dict

    def __getitem__(self, idx):
        image_id = list(self.image_info.keys())[idx]
        img_path = f"{self.root}/{self.image_info[image_id]['file_name']}"
        image = Image.open(img_path)
        

        
        # 獲取標注
        target = {}
        target['boxes'] = torch.as_tensor(
            [ann['bbox'] for ann in self.annotations[image_id]], 
            dtype=torch.float32
        )
        target['labels'] = torch.as_tensor(
            [ann['category_id'] for ann in self.annotations[image_id]], 
            dtype=torch.int64
        )
        target['image_id'] = torch.tensor([image_id])
        
        # return image, target

        # # 同步處理變換
        # transform_chain = ComposeWithBox([ 
        #     ResizeWithBox((224, 224)),
        #     T.ToTensor(),

        #     # RandomHorizontalFlipWithBox(p=0.5),
        #     # 添加其他需要同步處理的變換...

        #     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        
        # return transform_chain(image, target)
        

        image, target = ResizeWithBox((224, 224))(image, target)        # If need to validate transformation, comment this row
        
        # 數據預處理
        if self.transforms:
            image = self.transforms(image)
            
        return image, target
        

    def __len__(self):
        return len(self.image_info)
    
    def validate_transformation(self, dataset):
        import random
        from matplotlib import pyplot as plt
        # 加載原始樣本

        ind = random.randrange(0, len(dataset))
        image, target = dataset[ind]
        print(f'Original image size: {image.size}')
        print(f'Original target: {target}')
        orig_boxes = target['boxes'].clone()
        
        trfm_img, trfm_target = ResizeWithBox((224, 224))(image, target)

        # from torchvision.ops import box_convert
        # trfm_target = box_convert(trfm_target['boxes'], in_fmt='xyxy', out_fmt='xywh')
        
        print(f'Trfm image size: {trfm_img.size}')
        print(f'Trfm target: {trfm_target}')

        # 可視化驗證
        plt.figure(figsize=(12,6))
        
        # 原始圖像與框
        plt.subplot(121)
        or_img = T.ToTensor()(image)
        # or_img = image
        plt.imshow(or_img.permute(1,2,0).cpu().numpy())
        for box in orig_boxes:
            x, y, w, h = box
            # w = x2 - x1         # 寬度計算
            # h = y2 - y1
            plt.gca().add_patch(plt.Rectangle((x,y),w,h, fill=False, edgecolor='r'))
        
        # 變換後圖像與框
        plt.subplot(122)
        trfm_img_tensor = T.ToTensor()(trfm_img)
        # trfm_img_tensor = trfm_img
        plt.imshow(trfm_img_tensor.permute(1,2,0).cpu().numpy())
        for box in trfm_target['boxes']:
            # x, y, w, h = box
            # plt.gca().add_patch(plt.Rectangle((x,y),w,h, fill=False, edgecolor='b'))
            x1, y1, x2, y2 = box
            w = x2 - x1         # 寬度計算
            h = y2 - y1
            plt.gca().add_patch(plt.Rectangle((x1,y1),w,h, fill=False, edgecolor='r'))
        
        plt.show()

    
class ResizeWithBox(object):
    """同步調整圖像尺寸與邊界框坐標"""
    def __init__(self, size=(224, 224)):
        self.size = size  # (height, width)
    
    def __call__(self, image, target):
        # 原始尺寸
        if isinstance(image, Image.Image):  # PIL格式
            orig_w, orig_h = image.size
        else:  # Tensor格式
            orig_h, orig_w = image.shape[1], image.shape[2]
        
        # print(f'Original size: {orig_w}x{orig_h}')
        
        # 計算縮放比例
        scale_w = self.size[1] / orig_w
        scale_h = self.size[0] / orig_h
        
        # 調整圖像尺寸
        resized_image = F.resize(image, self.size)
        
        # print(f'Resized size: {resized_image}')
        
        # 調整邊界框坐標 (COCO格式: [x, y, w, h])
        boxes = target['boxes'].clone()
        
        # print(f'Original boxes: {boxes}')
        if boxes.numel() > 0:
            boxes[:, 0] *= scale_w  # x
            boxes[:, 1] *= scale_h  # y
            boxes[:, 2] *= scale_w  # w
            boxes[:, 3] *= scale_h  # h
            
        # print(f'Resized boxes: {boxes}')
        
        from torchvision.ops import box_convert
        boxes = box_convert(boxes, in_fmt='xywh', out_fmt='xyxy')
        
        # 更新目標數據
        target['boxes'] = boxes
        return resized_image, target

# class ComposeWithBox(torch.nn.Module):
#     """支持邊界框同步處理的組合變換"""
#     def __init__(self, transforms):
#         super().__init__()
#         self.transforms = transforms
    
#     def forward(self, image, target):
#         for t in self.transforms:
#             # if isinstance(t, (ResizeWithBox, RandomHorizontalFlipWithBox)):
#             if isinstance(t, ResizeWithBox):
#                 image, target = t(image, target)
#             else:
#                 image = t(image)
#         return image, target


if __name__ == '__main__':
    dataset = DigitDataset(
        root='dataset/train',
        annotation_path='dataset/train.json',
    )

    print(f'Dataset length: {len(dataset)}')

    print(dataset.validate_transformation(dataset))