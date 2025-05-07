
"""
Generate submission for competition from trained Cascade Mask R‑CNN.
"""
import argparse, json
from pathlib import Path
import cv2
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from dataset import register_cell_dataset
from config import *# DATA_ROOT, NUM_CLASSES, CASCADE_YAML
# from utils11.rle import encode_binary_mask
from utils import encode_mask


def setup():
    cfg=get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_YAML))
    cfg.MODEL.WEIGHTS='results/misc50/model_final.pth'#str(OUTPUT_DIR / 'model_final.pth')
    cfg.MODEL.ROI_HEADS.NUM_CLASSES=NUM_CLASSES
    cfg.INPUT.MIN_SIZE_TEST=0
    # cfg.INPUT.MAX_SIZE_TEST=1024
    # cfg.MODEL.DEVICE='cuda'
    cfg.freeze()
    return cfg


def _id_map():
    with open(DATA_ROOT/'test_image_name_to_ids.json') as f:
        return {d['file_name']:d['id'] for d in json.load(f)}


def _test_imgs():
    return sorted((DATA_ROOT/'test_release').glob('*.tif'))


def main():
    # register_cell_dataset()
    predictor=DefaultPredictor(setup())
    # id_map=_id_map()

    test_file_name_id = {}
    with open(DATA_ROOT/'test_image_name_to_ids.json') as f:
        for img_info in json.load(f):
            name = img_info['file_name']
            id = img_info['id']
            test_file_name_id[name] = id
        

    results=[]
    for img_path in list((DATA_ROOT/'test_release').glob('*.tif')):
        img = cv2.imread(str(img_path))
        h, w = img.shape[0], img.shape[1]
        
        output = predictor(img)['instances'].to('cpu')
        for box, mask, class_, score in zip(output.pred_boxes.tensor.numpy(),
                                       output.pred_masks.numpy(),
                                       output.pred_classes.numpy(),
                                       output.scores.numpy()):
            # x1,y1,x2,y2=box.tolist()
            # rle=encode_binary_mask(mask)
            # print(f'box: {type(box.tolist()[0])}')
            # print(f'mask: {mask}')
            # print(f'class_: {type(class_)}')
            # print(f'score: {type(score)}')
            rle = encode_mask(mask)
            results.append({
                'image_id': int(test_file_name_id[img_path.name]),
                'bbox': box.tolist(),
                'score': float(score),
                'category_id': int(class_ + 1),
                # 'segmentation': {'size':[int(h),int(w)],'counts':rle['counts']}
                'segmentation': rle
            })

    # out_zip=Path(args.output)
    # out_zip.parent.mkdir(parents=True,exist_ok=True)
    # json_path=out_zip.parent/'test-results.json'
    json_path = './test-results.json'
    with open(json_path,'w') as f: json.dump(results,f)
    # import zipfile
    # with zipfile.ZipFile(out_zip,'w',zipfile.ZIP_DEFLATED) as zf:
    #     zf.write(json_path,arcname='test-results.json')
    # print(f'Submission written → {out_zip}')


if __name__=='__main__':
    # ap=argparse.ArgumentParser()
    # ap.add_argument('--weights',required=True)
    # # ap.add_argument('--output',default='submission.zip')
    # main(ap.parse_args())
    main()
