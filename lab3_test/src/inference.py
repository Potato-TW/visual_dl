import json
import cv2

from config import *
from utils import encode_mask

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

def setup():
    cfg=get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_YAML))
    cfg.MODEL.WEIGHTS='log/model_final.pth'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES=NUM_CLASSES
    cfg.INPUT.MIN_SIZE_TEST=0
    # cfg.INPUT.MAX_SIZE_TEST=1024
    cfg.freeze()
    return cfg


if __name__=='__main__':
    predictor=DefaultPredictor(setup())

    test_file_name_id = {}
    with open(DATA_ROOT/'test_image_name_to_ids.json') as f:
        for img_info in json.load(f):
            name = img_info['file_name']
            id = img_info['id']
            test_file_name_id[name] = id
        

    results=[]
    for img_path in list((DATA_ROOT/'test_release').glob('*.tif')):
        img = cv2.imread(str(img_path))
        
        output = predictor(img)['instances'].to('cpu')
        for box, mask, class_, score in zip(output.pred_boxes.tensor.numpy(),
                                            output.pred_masks.numpy(),
                                            output.pred_classes.numpy(),
                                            output.scores.numpy()):
            rle = encode_mask(mask)
            results.append({
                'image_id': int(test_file_name_id[img_path.name]),
                'bbox': box.tolist(),
                'score': float(score),
                'category_id': int(class_ + 1),
                'segmentation': rle
            })

    json_path = './test-results.json'
    with open(json_path, 'w') as f: 
        json.dump(results, f)