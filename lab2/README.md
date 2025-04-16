# NYCU Visual Recognitionusing Deep Learning 2025 Spring LAB 2

StudentID: 110550130<br>
Name: 劉秉驊<br>

## Introduction

In this lab, we implement Fast RCNN with different backbones to recognize digits in images.<br>
We introduce Mobile V2 and V3, ResNet50 with several training strategies.<br>
Finally, generate labels and boxes as json file, and also recognize the while number in images.<br>

## How to install
1. Google colab<br>
    1. I use A100 GPU to train model.
    1. No need to worry about environment, run all.

1. Run locally<br>
    1. Import env.yml in conda first.
    1. Activate conda virtual environment.

To change model settings:<br>
1. train.py<br>
We provide 3 backbones:<br>
    ```python
    # MobileNet v2
    def build_model(num_classes=11):
        ...
    # MobileNet v3
    def build_model_v3(num_classes=11):
        ...
    # ResNet 50
    def build_model_resnet50(num_classes=11):
        ...
    ```

    Don't forget to change model variable here.<br>
    And change epochs as hyperparameter.<br>

    ![train model change](./img/trainPY_epoch.png)

    We also provide several optimizer / scheduler combination.<br>

    ```python
    def first_version_v2(model):
        ...

    def gpt_recommend_v2_speedup(model, train_data_loader):
        ... 

    def reference_optim_scheduler(model):
        ...

    def v2_reconstrust(model):
        ...

    optimizer, lr_scheduler = first_version_v2(model)
    # optimizer, lr_scheduler = gpt_recommend_v2_speedup(model, train_data_loader)
    # optimizer, lr_scheduler = reference_optim_scheduler(model)
    ```

    After settings, type command below:
    ```bash
    python3 src/train.py
    ```

1. test.py<br>
You need to change model backbone to load model weights.<br><br>
For example, 
    1. if we use MobileNet v2 to train, we need to change to 
        ```python 
        model = load_model(ckpt_path, 11).to(device) 
        ```
    1. if we use ResNet50 to train, we need to change to 
        ```python
        model = load_model_resnet50(ckpt_path, 11).to(device)
    And don't forget to change checkpoint path.<br>

    ![test change](./img/test_model_change.png)

    After settings, type command below:
    ```bash
    python3 src/test.py
    ```


## Performance snapshot
We use MobileNet v2 to recognize digits.<br>
And if you use AdamW + Cosine Annealing LR as well, you can probabiliy get the below result.<br>

But we don't guarantee the results of the other settings.<br>

![result V2](./img/train_loss_first_version_v2.png)