import json
import os
import pathlib
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import transforms as T
# from torchvision.transforms import v2 as T

import numpy as np
import random

class HW4_REALSE_DATASET(Dataset):
    def __init__(self, root_path: Path, mode, output_img_size, shuffle_list: list=None):
        """
        參數說明：
        json_path: 包含圖像路徑與標籤的JSON文件路徑
        objects_map_path: 物體名稱到索引的映射文件
        img_dir: 圖像存儲基礎路徑
        img_size: 輸出圖像尺寸
        """
        self.root_path = root_path
        self.mode = mode
        assert mode in ['train', 'valid', 'test'], f"{mode} should be 'train' or 'test'."
        self.output_img_size = output_img_size
        self.shuffle_list = shuffle_list
        self.transform = self.get_trfm()

        # self.img_dir_path = self.root_path / mode if self.mode in ['train', 'test'] else self.root_path / 'train'

        if self.mode in ['train', 'valid']:
            self.img_dir_path = self.root_path / 'train'

            self.clean_imgs = self.get_img_list(self.img_dir_path / 'clean')
            self.degraded_imgs = self.get_img_list(self.img_dir_path / 'degraded')
        else:
            self.img_dir_path = self.root_path / 'test'

            self.degraded_imgs = self.get_img_list(self.img_dir_path / 'degraded')

    def get_img_list(self, img_dir: Path):
        a = sorted(img_dir.glob('*.png'))

        if self.shuffle_list is None:
            return a

        if self.mode == 'train':
            return [img for img, keep in zip(a, self.shuffle_list) if keep]
        elif self.mode == 'valid':
            return [img for img, keep in zip(a, self.shuffle_list) if not keep]
        else:
            return a

    def get_trfm(self):
        if self.mode in ['train']:
            tfrm = T.Compose([
                T.RandomCrop(self.output_img_size),
                # T.RandomHorizontalFlip(p=0.5),
                # T.RandomVerticalFlip(p=0.5),
                # T.RandomRotation(30),
                # T.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.3, hue=0.1),
                T.ToTensor(),
                # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif self.mode in ['valid']:
            tfrm = T.Compose([
                # T.Resize((self.output_img_size, self.output_img_size)),
                T.ToTensor(),
                # T.Normalize(mean = [0.485, 0.456, 0.406],
                #             std = [0.229, 0.224, 0.225])
            ])
        else:
            tfrm = T.Compose([
                # T.Resize((self.output_img_size, self.output_img_size)),
                T.ToTensor(),
                # T.Normalize(mean = [0.485, 0.456, 0.406],
                #             std = [0.229, 0.224, 0.225])
            ])
        
        return tfrm

    def __len__(self):
        return len(self.degraded_imgs)

    def __getitem__(self, idx):
        seed = np.random.randint(2147483647) 
        if self.mode in ['train', 'valid']:
            degraded_img_path = self.degraded_imgs[idx]
            type_, num_ = degraded_img_path.stem.split('-')

            clean_img_path = self.img_dir_path / 'clean' / f'{type_}_clean-{num_}.png'
            if clean_img_path not in self.clean_imgs:
                raise ValueError('Not found clean image')

            degraded_img = Image.open(degraded_img_path).convert('RGB')
            clean_img = Image.open(clean_img_path).convert('RGB')
            
            if self.transform:
                random.seed(seed)
                torch.manual_seed(seed)
                degraded_img = self.transform(degraded_img)
                random.seed(seed)
                torch.manual_seed(seed)
                clean_img = self.transform(clean_img)
                
            return degraded_img, clean_img
        else:
            degraded_img_path = self.degraded_imgs[idx]

            degraded_img = Image.open(degraded_img_path).convert('RGB')
            if self.transform:
                degraded_img = self.transform(degraded_img)

            img_name = degraded_img_path.stem
            return img_name, degraded_img

if __name__ == "__main__":
    img_size = 64

    total = 3200
    true_count = int(total * 0.8)  # 2560
    false_count = total - true_count  # 640

    # 創建初始列表
    bool_list = [True] * true_count + [False] * false_count

    # 隨機打亂順序
    random.shuffle(bool_list)

    train_set = HW4_REALSE_DATASET(root_path=Path('./hw4_realse_dataset'), mode='train', output_img_size=img_size, shuffle_list=bool_list)
    val_set = HW4_REALSE_DATASET(root_path=Path('./hw4_realse_dataset'), mode='valid', output_img_size=img_size, shuffle_list=bool_list)
    test_set = HW4_REALSE_DATASET(root_path=Path('./hw4_realse_dataset'), mode='test', output_img_size=img_size, shuffle_list=bool_list)

    img, c_img = train_set[0]
    # print(img, img.shape)

    import numpy as np
    pil_image = T.ToPILImage()(img)
    c_pil_image = T.ToPILImage()(c_img)
    pil_image.save('de.png')
    c_pil_image.save('cl.png')