import os
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

# from torchvision.transforms import v2 as T
from torchvision.utils import make_grid, save_image

from diffusers import DDPMScheduler

from dataset import HW4_REALSE_DATASET
# from ddpm import CondDDPM
# from evaluator import evaluation_model
from model import PromptIR

import wandb
from matplotlib import pyplot as plt
import pathlib
from pathlib import Path

class Training():
    def __init__(self, args):
        self.epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.dataset_path = args.dataset_path
        self.img_size = args.output_img_size
        self.num_workers = args.num_workers
        self.device = args.device
        self.train_ratio = args.train_ratio
        self.seed = args.seed
        self.lr = args.lr
        self.ckpt_dir = args.save_ckpt_dir
        self.save_img_dir = args.save_img_dir
        self.save_frequency = args.save_frequency

        
        shuffle_list = self.generate_shuffle_list()
        self.train_set = HW4_REALSE_DATASET(root_path=Path(self.dataset_path),
                                            mode="train", 
                                            output_img_size=self.img_size,
                                            shuffle_list=shuffle_list)
        
        self.train_loader = DataLoader(self.train_set, 
                                       batch_size=self.batch_size, 
                                       shuffle=True, 
                                       num_workers=self.num_workers)
        
        self.val_set = HW4_REALSE_DATASET(root_path=Path(self.dataset_path),
                                            mode="valid", 
                                            output_img_size=self.img_size,
                                            shuffle_list=shuffle_list)
        
        self.val_loader = DataLoader(self.val_set, 
                                       batch_size=self.batch_size, 
                                       shuffle=False, 
                                       num_workers=self.num_workers)
        
        # self.test_set = HW4_REALSE_DATASET(root_path=Path(self.dataset_path),
        #                                    mode="test", 
        #                                    output_img_size=self.img_size,
        #                                    shuffle_list=shuffle_list)

        # self.test_loader = DataLoader(self.test_set, 
        #                                batch_size=self.batch_size,
        #                                shuffle=False, 
        #                                num_workers=self.num_workers)
        

        
        self.model = PromptIR(decoder=True).to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                              T_max=len(self.train_loader) * self.epochs,
                                                              last_epoch=-1,
                                                              eta_min=1e-9)
        # self.lr_scheduler = torch.optim.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer=self.optimizer,
        #                                                                            warmup_epochs=15,
        #                                                                            max_epochs=150)
        # self.loss_function = nn.MSELoss()
        self.loss_function = nn.L1Loss()

    def generate_shuffle_list(self):
        random.seed(self.seed)

        total = 3200
        true_count = int(total * self.train_ratio)  # 2560
        false_count = total - true_count  # 640

        # 創建初始列表
        bool_list = [True] * true_count + [False] * false_count

        # 隨機打亂順序
        random.shuffle(bool_list)

        return bool_list

    def save_ckpt(self, epoch):
        save_path = os.path.join(self.ckpt_dir, f'ckpt_{epoch}.pth')
        print(f'Saving ckpt to {save_path}')
        torch.save({'model': self.model.state_dict(), 
                    'optimizer': self.optimizer.state_dict()}, 
                    save_path)

    @torch.no_grad()
    def eval(self, dataloader=None):
        self.model.eval()

        eval_loss = []
        progress_bar = tqdm(dataloader, desc='Eval', ncols=100)
        for img, gt in progress_bar:
            img, gt = img.to(self.device), gt.to(self.device)

            output = self.model(img)

            loss = self.loss_function(output, gt)

            eval_loss.append(loss.item())
            progress_bar.set_postfix({"loss": np.mean(eval_loss)})
            progress_bar.update()

        return np.mean(eval_loss)

    def train_one_epoch(self):
        self.model.train()
        
        train_loss = []
        progress_bar = tqdm(self.train_loader, desc=f'Training', ncols=100)
        for i, (img, gt) in enumerate(progress_bar):
            img, gt = img.to(self.device), gt.to(self.device)

            output = self.model(img)
            
            loss = self.loss_function(output, gt)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            
            train_loss.append(loss.item())
            progress_bar.set_postfix({'loss': np.mean(train_loss)})
            progress_bar.update()
            
        return np.mean(train_loss)

    def run(self):
        for epoch in tqdm(range(self.epochs), desc="Epochs", ncols=100):
            loss = self.train_one_epoch()
            wandb.log({
                "Epoch": epoch,
                "Train loss": loss,
            })

            eval_loss = self.eval(dataloader=self.val_loader)
            wandb.log({
                "Epoch": epoch,
                "Eval loss": eval_loss,
            })

            if epoch % self.save_frequency == 0:
                self.save_ckpt(epoch)

        self.save_ckpt(-1)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="lab4")
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--output-img-size', type=int, default=128)
    parser.add_argument('--dataset_path', '-ds', type=str, default='./hw4_realse_dataset')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-ckpt-dir', type=str, default='./ckpts')
    parser.add_argument('--save-img-dir', type=str, default='./img')
    parser.add_argument('--save-frequency', type=int, default=5)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    args = parser.parse_args()


    os.makedirs(args.save_ckpt_dir, exist_ok=True)
    os.makedirs(args.save_img_dir, exist_ok=True)
    wandb.init(project="VDL-lab4", name=args.wandb_run_name, save_code=True)
    
    process = Training(args)

    process.run()