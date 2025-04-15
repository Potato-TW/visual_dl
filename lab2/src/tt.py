def show_task1_with_triangle():
    import json
    import random
    import matplotlib.pyplot as plt
    from PIL import Image

    with open('./result/task1/pred.json', 'r') as f:
        coco_results = json.load(f)        
        
    ind = random.randrange(0, len(coco_results))
    img_meta = coco_results[ind]
    img_id = img_meta['image_id']
    img_box = img_meta['bbox']

    img = Image.open(f'./dataset/test/{int(img_id)}.png')

    # 可視化驗證
    plt.figure(figsize=(12,6))

    # 原始圖像與框
    plt.subplot(121)
    plt.imshow(img)
    x, y, w, h = img_box
    plt.gca().add_patch(plt.Rectangle((x,y),w,h, fill=False, edgecolor='r'))

    plt.show()
    
    
def test_pandas():
    import pandas as pd

    # 創建數據
    data = {
        "image_id": range(1, 13069),  # 生成1~13068的連續ID
        "pred_label": -1              # 全部填充-1
    }

    # 構建DataFrame
    df = pd.DataFrame(data)

    # 驗證結構
    print(f"數據維度: {df.shape}")   # 應輸出 (13068, 2)
    print("\n前5行示例:")
    print(df.head())
    print("\n後5行示例:")
    print(df.tail())

    # 將DataFrame保存為CSV文件
    df.to_csv('./result/task2/pred.csv', index=False)
    
def test_argsort():
    import numpy as np
    a = np.array([[152.7115,  80.3657, 172.8028, 185.9408],
            [137.5092,  73.9137, 154.3575, 187.1175]])
    b = np.array([3, 2])

    print(a.shape)
    print(b.shape)

    sort_idx = np.argsort(b)  # 输出 [1, 0]

    # 应用排序
    a_sorted = a[sort_idx]
    b_sorted = b[sort_idx]

    print(a_sorted)
    print(b_sorted)
    
def task2_do(output, task2, image_id):
    """
    將模型輸出與預設DataFrame結合
    """
    # 取出當前圖像的預測結果
    boxes = output['boxes'].cpu().detach().numpy()  # (N,4) tensor轉numpy
    # scores = output['scores'].cpu().detach().numpy().astype(float)  # (N,)
    labels = output['labels'].cpu().detach().numpy()  # (N,)
    
    value = []
    
    
    # 更新對應的行
    for i in range(boxes.shape[0]):
        if scores[i] < 0.5:
            continue
        
        task2.loc[task2['image_id'] == int(image_id.item()), 'pred_label'] = int(labels[i])
    
    return task2

# import torch

# tmp = {'boxes': tensor([[152.7115,  80.3657, 172.8028, 185.9408],
#         [137.5092,  73.9137, 154.3575, 187.1175]], device='cuda:0'), 'labels': tensor([3, 2], device='cuda:0'), 'scores': tensor([0.9968, 0.9428], device='cuda:0')}

import numpy as np 

a = np.array([])
a.append(1)

print(a)