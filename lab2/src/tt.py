import json

with open('dataset/train.json') as f:
    coco_data = json.load(f)
print(coco_data.keys())
print(coco_data['images'][0])
print(coco_data['annotations'][0])
print(coco_data['categories'][0])