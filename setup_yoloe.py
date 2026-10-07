"""YOLOE pretrained text-prompt setup, no model training."""
import os
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.yolo'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mpl'))
os.environ.setdefault('TORCH_HOME',str(ROOT/'models/cache'))
from ultralytics import YOLOE, settings
import yaml
settings.update({'weights_dir':str(ROOT/'models'),'sync':False})
names=yaml.safe_load((ROOT/'data/raw/data.yaml').read_text())['names']
model=YOLOE(str(ROOT/'models/yoloe-11s-seg.pt'))
print('Encoding YOLOE vocabulary, no training',flush=True)
model.set_classes(names,model.get_text_pe(names))
model.model.clip_model=None
output=ROOT/'models/yoloe-11s-produce.pt'
model.save(str(output))
selection={'run_id':'yoloe11s-pretrained-47','model':'YOLOE 11 small + MobileCLIP vocabulary',
           'weights':str(output.relative_to(ROOT)).replace('\\','/'),'mode':'pretrained_zero_shot',
           'classes':names,'supported_classes':names,'class_map':{n:n for n in names},
           'confidence_threshold':0.25,'iou_threshold':0.45,'imgsz':640,'trained_locally':False,
           'weights_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
           'source':'https://docs.ultralytics.com/models/yoloe',
           'limitations':'47 candidate classes, not 47 validated accurate classes. Closely related produce can be confused.'}
(ROOT/'artifacts/yoloe_config.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf-8')
print('YOLOE ready',flush=True)
