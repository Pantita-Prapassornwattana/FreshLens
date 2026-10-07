"""Map exact/synonym labels of author's pretrained63-class produce detector."""
import os
import json
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.yolo'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mpl'))
from ultralytics import YOLO
import yaml
names=yaml.safe_load((ROOT/'data/raw/data.yaml').read_text())['names']
weights=ROOT/'models/fruit_vegetable_yolov8m.pt'
model=YOLO(str(weights))
aliases={'bell pepper/capsicum':'bell pepper','edible corn/corn/maize':'corn','cucumber/cuke':'cucumber',
         'eggplant/aubergine':'eggplant','garlic/ail':'garlic','green onion/spring onion/scallion':'green onion',
         'kiwi fruit':'kiwi','mandarin orange':'mandarin','orange/orange fruit':'orange','pea/pea food':'pea',
         'radish/daikon':'radish','zucchini/courgette':'vegetable marrow',
         'chili/chili vegetable/chili pepper/chili pepper vegetable/chilli/chilli vegetable/chilly/chilly':'hot pepper'}
class_map={n:(n if n in names else aliases[n]) for n in model.names.values() if n in names or n in aliases}
supported=sorted(set(class_map.values()))
config={'run_id':'author-pretrained-yolov8m-produce','model':'YOLOv8m Fruits & Vegetables (author pretrained)',
        'weights':str(weights.relative_to(ROOT)).replace('\\','/'),'mode':'pretrained_domain_model',
        'classes':names,'supported_classes':supported,'class_map':class_map,'confidence_threshold':0.25,
        'iou_threshold':0.45,'imgsz':640,'trained_locally':False,
        'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
        'source':'https://github.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset',
        'excluded_model_labels':['Strawberry','Tomato'],
        'unsupported_classes':sorted(set(names)-set(supported)),
        'limitations':'Author-trained model has63 labels, mapped to dataset synonyms only. Duplicate uppercase labels excluded following author note. Missing dataset classes cannot be identified by this model.'}
(ROOT/'artifacts/domain_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'supported':supported,'unsupported':config['unsupported_classes']}))
