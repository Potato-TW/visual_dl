import os
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from torchvision.transforms import transforms as T
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

class Tester():
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.dataset_path = args.dataset_path
        self.img_size = args.output_img_size
        self.num_workers = args.num_workers
        self.device = args.device
        self.seed = args.seed
        self.ckpt_path = args.ckpt_path
        self.save_img_dir = args.save_img_dir
        
        self.test_set = HW4_REALSE_DATASET(root_path=Path(self.dataset_path),
                                           mode="test", 
                                           output_img_size=self.img_size,)

        self.test_loader = DataLoader(self.test_set, 
                                       batch_size=1,#self.batch_size,
                                       shuffle=False, 
                                       num_workers=self.num_workers)
        

        
        self.model = PromptIR(decoder=True).to(self.device)
        self.load_ckpt(self.ckpt_path)


    def load_ckpt(self, ckpt_path):
        self.model.load_state_dict(torch.load(ckpt_path)['model'])
        print(f'Loading ckpt to {ckpt_path}')
        
    def save_img(self, img_name, output: torch.tensor):
        img_name = img_name[0]
        output = output.detach().cpu()[0]

        # mean = torch.tensor([0.485, 0.456, 0.406])
        # std = torch.tensor([0.229, 0.224, 0.225])
        # output = output * std[:, None, None] + mean[:, None, None]  # 維度擴展匹配

        output = output.numpy()

        # output = torch.clamp(output * 255, 0, 255).byte()
        ar = np.clip(output * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)

        from PIL import Image
        pil_image = Image.fromarray(ar)

        # pil_image = T.ToPILImage()(output.squeeze())
        # print(pil_image)

        pil_image.save(f"{self.save_img_dir}/{img_name}.png")


    @torch.no_grad()
    def test(self):
        self.model.eval()

        # eval_loss = []
        progress_bar = tqdm(self.test_loader, desc='Eval', ncols=100)
        for img_name, img in progress_bar:
            img = img.to(self.device)

            output = self.model(img)

            self.save_img(img_name, output)

            progress_bar.update()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--output-img-size', type=int, default=64)
    parser.add_argument('--dataset_path', '-ds', type=str, default='./hw4_realse_dataset')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ckpt-path', type=str, default='/home/bhg/visual_dl/lab4/ckpts/ckpt_150.pth')
    parser.add_argument('--save-img-dir', type=str, default='./inf_img')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    os.makedirs(args.save_img_dir, exist_ok=True)
    
    process = Tester(args)

    process.test()