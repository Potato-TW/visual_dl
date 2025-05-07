
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
from config import DATA_ROOT, NUM_CLASSES, CASCADE_YAML
from utils.rle import encode_binary_mask


def _cfg(weights):
    cfg=get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CASCADE_YAML))
    cfg.MODEL.WEIGHTS=weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES=NUM_CLASSES
    cfg.INPUT.MIN_SIZE_TEST=0
    cfg.INPUT.MAX_SIZE_TEST=1024
    cfg.MODEL.DEVICE='cuda'
    cfg.freeze()
    return cfg


def _id_map():
    with open(DATA_ROOT/'test_image_name_to_ids.json') as f:
        return {d['file_name']:d['id'] for d in json.load(f)}


def _test_imgs():
    return sorted((DATA_ROOT/'test_release').glob('*.tif'))


def main(args):
    register_cell_dataset()
    predictor=DefaultPredictor(_cfg(args.weights))
    id_map=_id_map()
    results=[]
    for pth in tqdm(_test_imgs()):
        img=cv2.imread(str(pth))
        h,w=img.shape[:2]
        inst=predictor(img)['instances'].to('cpu')
        for box,mask,cls,score in zip(inst.pred_boxes.tensor.numpy(),
                                       inst.pred_masks.numpy(),
                                       inst.pred_classes.numpy(),
                                       inst.scores.numpy()):
            x1,y1,x2,y2=box.tolist()
            rle=encode_binary_mask(mask)
            results.append({
                'image_id':int(id_map[pth.name]),
                'bbox':[float(x1),float(y1),float(x2),float(y2)],
                'score':float(score),
                'category_id':int(cls)+1,
                'segmentation':{'size':[int(h),int(w)],'counts':rle['counts']}
            })
    out_zip=Path(args.output)
    out_zip.parent.mkdir(parents=True,exist_ok=True)
    json_path=out_zip.parent/'test-results.json'
    with open(json_path,'w') as f: json.dump(results,f)
    import zipfile
    with zipfile.ZipFile(out_zip,'w',zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path,arcname='test-results.json')
    print(f'Submission written → {out_zip}')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--weights',required=True)
    ap.add_argument('--output',default='submission.zip')
    main(ap.parse_args())
