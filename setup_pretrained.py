"""Prepare fixed vocabulary inference checkpoints. No optimizer, no training."""
import hashlib
import json
import os
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.yolo'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mpl'))
os.environ.setdefault('TORCH_HOME',str(ROOT/'models/cache'))
from ultralytics import YOLOWorld, settings
import yaml

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['s','m'],default='s')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    variant={'s':'small','m':'medium'}[args.variant]
    settings.update({'weights_dir':str(ROOT/'models'),'sync':False})
    names=yaml.safe_load((ROOT/'data/raw/data.yaml').read_text())['names']
    output=ROOT/f'models/yolov8{args.variant}-worldv2-produce.pt'
    if not output.exists():
        model=YOLOWorld(str(ROOT/f'models/yolov8{args.variant}-worldv2.pt'))
        print('Encoding fixed produce vocabulary with pretrained CLIP; no training',flush=True)
        model.set_classes(names)
        model.model.clip_model=None
        model.save(str(output))
    selection={'run_id':f'worldv2-{variant}-pretrained-47','model':f'YOLO-World v2 {variant} + CLIP vocabulary',
               'weights':str(output.relative_to(ROOT)).replace('\\','/'),'mode':'pretrained_zero_shot',
               'classes':names,'supported_classes':names,'class_map':{n:n for n in names},
               'confidence_threshold':0.25,'iou_threshold':0.45,'imgsz':640,'trained_locally':False,
               'weights_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
               'source':'https://docs.ultralytics.com/models/yolo-world',
               'limitations':'Vocabulary has 47 candidate classes. This does not imply all 47 are accurate. Similar produce may be confused.'}
    (ROOT/'artifacts').mkdir(exist_ok=True)
    for name in ([f'world_{variant}_config.json'] if args.prepare_only else [f'world_{variant}_config.json','selected_model.json']):
        (ROOT/'artifacts'/name).write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf-8')
    print('World ready: fixed47 vocabulary, offline checkpoint saved',flush=True)

if __name__=='__main__':main()
