import pathlib

a = pathlib.Path('hw4_realse_dataset/train/degraded')

# for i in a.glob('*.png'):
#     print(i)

# d = sorted(a.glob('*.png'))
# print(d[0].stem)

# import cv2
# import numpy as np
# e = cv2.imread('img/99_clean.png')
# print(e, np.array(e).shape)

# b = pathlib.Path('hw4_realse_dataset/train/degraded/snow-541.png')
# t, n = b.stem.split('-')
# print(t, n)

# import torch 

# f = torch.load('ckpts_1/ckpt_-1.pth')
# print(f['optimizer'])

import numpy as np

f = np.load('./pred.npz')
print(sorted(f.files))
