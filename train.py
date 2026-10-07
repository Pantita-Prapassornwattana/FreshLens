"""Reproducible learning-curve/model comparison; validation selects, test stays sealed."""
import argparse
import csv
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.yolo'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mpl'))
from ultralytics import YOLO
import torch
import yaml

def save(path, obj):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2),encoding='utf-8')
    temporary.replace(path)

def metric(result):
    return {'precision':float(result.box.mp),'recall':float(result.box.mr),
            'map50':float(result.box.map50),'map50_95':float(result.box.map),
            'per_class_map50_95':{str(result.names[int(i)]):float(v) for i,v in zip(result.box.ap_class_index,result.box.ap)}}

def main(args):
    art=ROOT/'artifacts';art.mkdir(exist_ok=True)
    audit=json.loads((art/'dataset_audit.json').read_text(encoding='utf-8'))
    log=art/'experiments.json'
    records=json.loads(log.read_text(encoding='utf-8')) if log.exists() else []
    if args.final_test:
        selection_path=art/'selected_model.json'
        if not selection_path.exists():
            raise ValueError('Run experiments first; selection is based on validation only')
        if (art/'final_test.json').exists():
            raise ValueError('Final test already recorded. Do not tune against this test set.')
        selected=json.loads(selection_path.read_text(encoding='utf-8'))
        if selected['manifest_sha256']!=audit['manifest_sha256']:
            raise ValueError('Dataset changed after model selection')
        if hashlib.sha256((ROOT/selected['weights']).read_bytes()).hexdigest()!=selected['weights_sha256']:
            raise ValueError('Selected checkpoint changed after model selection')
        result=YOLO(str(ROOT/selected['weights'])).val(data=str(ROOT/'data/prepared/data_100.yaml'),split='test',device=args.device,imgsz=640,batch=args.batch,workers=0,project=str(ROOT/'runs'),name='final_test')
        save(art/'final_test.json',{'run_id':selected['run_id'],'metrics':metric(result),'time':datetime.now(timezone.utc).isoformat(),'manifest_sha256':audit['manifest_sha256']})
        return
    if (art/'final_test.json').exists():
        raise ValueError('Final test has been opened; use a new independent held-out dataset for further tuning.')
    for model_name in args.models:
        for fraction in args.fractions:
            data=ROOT/f'data/prepared/data_{fraction}.yaml'
            spec=yaml.safe_load(data.read_text(encoding='utf-8'))
            count=len(Path(spec['train']).read_text().splitlines())
            run_id=f'{Path(model_name).stem}_{fraction}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")}'
            record={'run_id':run_id,'model':model_name,'train_images':count,'fraction':fraction,'requested_epochs':args.epochs,
                    'status':'running','manifest_sha256':audit['manifest_sha256'],'seed':42,'start':datetime.now(timezone.utc).isoformat(),
                    'environment':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.cuda.is_available()},
                    'settings':{'imgsz':640,'batch':args.batch,'patience':10,'weight_decay':0.0005,'device':args.device}}
            records.append(record);save(log,records)
            start=time.perf_counter()
            try:
                # Each experiment starts from the same pretrained checkpoint, never the prior run.
                model=YOLO(model_name)
                def progress(trainer):
                    record['completed_epochs']=int(trainer.epoch)+1
                    record['seconds']=round(time.perf_counter()-start,1)
                    save(log,records)
                model.add_callback('on_fit_epoch_end',progress)
                model.train(data=str(data),epochs=args.epochs,imgsz=640,batch=args.batch,device=args.device,workers=0,
                            seed=42,deterministic=True,patience=10,weight_decay=0.0005,close_mosaic=10,
                            project=str(ROOT/'runs'),name=run_id,exist_ok=False,cache=False)
                directory=Path(model.trainer.save_dir)
                weights=directory/'weights/best.pt'
                best=YOLO(str(weights))
                validation=best.val(data=str(data),split='val',imgsz=640,batch=args.batch,device=args.device,workers=0,
                                    project=str(ROOT/'runs'),name=run_id+'_validation')
                rows=list(csv.DictReader((directory/'results.csv').open(encoding='utf-8')))
                record.update(status='completed',completed_epochs=len(rows),validation=metric(validation),
                              weights=str(weights.relative_to(ROOT)).replace('\\','/'),seconds=round(time.perf_counter()-start,1),
                              weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest())
            except Exception as exc:
                record.update(status='failed',error=str(exc),seconds=round(time.perf_counter()-start,1))
                save(log,records)
                raise
            save(log,records)
            eligible=[r for r in records if r['status']=='completed' and r['manifest_sha256']==audit['manifest_sha256']]
            selected=max(eligible,key=lambda r:r['validation']['map50_95'])
            save(art/'selected_model.json',{k:selected[k] for k in ('run_id','model','weights','weights_sha256','manifest_sha256','validation')})
    eligible=[r for r in records if r['status']=='completed' and r['manifest_sha256']==audit['manifest_sha256']]
    selected=max(eligible,key=lambda r:r['validation']['map50_95'])
    save(art/'selected_model.json',{k:selected[k] for k in ('run_id','model','weights','weights_sha256','manifest_sha256','validation')})
    print('Selected by validation:',selected['run_id'],'Run --final-test once after freezing all choices.')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--models',nargs='+',default=['models/yolo11n.pt','models/yolov8n.pt'])
    p.add_argument('--fractions',nargs='+',type=int,choices=[10,30,100],default=[10,30,100])
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--device',default='0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--final-test',action='store_true')
    main(p.parse_args())
