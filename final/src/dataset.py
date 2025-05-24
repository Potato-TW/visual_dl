import argparse
import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def split_dataset(args):
    # 初始化路徑
    src_img_dir = Path(args.src_images)
    src_label_dir = Path(args.src_labels)
    dst_root = Path(args.dst_root)

    # 創建目標目錄結構
    (dst_root/'images/train').mkdir(parents=True, exist_ok=True)
    (dst_root/'images/val').mkdir(parents=True, exist_ok=True)
    (dst_root/'labels/train').mkdir(parents=True, exist_ok=True)
    (dst_root/'labels/val').mkdir(parents=True, exist_ok=True)

    # 獲取有效文件列表（同時存在圖片和標籤）
    valid_files = []
    for img_path in src_img_dir.glob('*.*'):
        label_path = src_label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            valid_files.append((img_path, label_path))

    # 隨機打亂並分割
    random.seed(args.seed)
    random.shuffle(valid_files)
    split_idx = int(len(valid_files) * args.ratio)
    train_files = valid_files[:split_idx]
    val_files = valid_files[split_idx:]

    # 複製文件函數
    def copy_files(files, split_type):
        for img_src, label_src in tqdm(files, desc=f'Processing {split_type}'):
            # 複製圖片
            img_dst = dst_root/'images'/split_type/img_src.name
            if args.copy_or_move == 'copy':
                shutil.copy(img_src, img_dst)
            else:
                shutil.move(img_src, img_dst)

            # 複製標籤
            label_dst = dst_root/'labels'/split_type/label_src.name
            if args.copy_or_move == 'copy':
                shutil.copy(label_src, label_dst)
            else:
                shutil.move(label_src, label_dst)

    copy_files(train_files, 'train')
    copy_files(val_files, 'val')

    # 生成統計報告
    print(f"\n{' Split Result ':=^40}")
    print(f"Total valid pairs: {len(valid_files)}")
    print(f"Train set: {len(train_files)} ({len(train_files)/len(valid_files):.1%})")
    print(f"Val set: {len(val_files)} ({len(val_files)/len(valid_files):.1%})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO Dataset Splitter')
    parser.add_argument('--src_images', type=str, default='dataset/images/trainval',
                        help='原始圖片路徑')
    parser.add_argument('--src_labels', type=str, default='dataset/labels/trainval',
                        help='原始標籤路徑')
    parser.add_argument('--dst_root', type=str, default='dataset',
                        help='輸出根目錄')
    parser.add_argument('--ratio', type=float, default=0.8,
                        help='訓練集比例 (0-1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子')
    parser.add_argument('--copy_or_move', type=str, default='copy', 
                        choices=['copy', 'move'],
                        help='複製或移動原始文件')
    args = parser.parse_args()

    split_dataset(args)