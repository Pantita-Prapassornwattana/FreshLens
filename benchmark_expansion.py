"""Evaluate frozen validation/test inputs without adjusting any model weights."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.yolo'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mpl'))
from PIL import Image,ImageOps
from produce_detector import ProduceDetector
from tests.evaluate_pretrained import match,metrics

def summarize(rows,names):
    counts=defaultdict(Counter)
    for row in rows:
        hits=match(row['detections'],row['truth'])
        hit_preds={h['prediction_index'] for h in hits}
        hit_truth={h['truth_index'] for h in hits}
        row.update(matches=hits,tp=len(hits),fp=len(row['detections'])-len(hits),fn=len(row['truth'])-len(hits))
        for i,p in enumerate(row['detections']):counts[p['class']]['tp' if i in hit_preds else 'fp']+=1
        for i,t in enumerate(row['truth']):
            if i not in hit_truth:counts[t['class']]['fn']+=1
    per={n:metrics(counts[n]['tp'],counts[n]['fp'],counts[n]['fn']) for n in names}
    a=metrics(sum(v['tp'] for v in per.values()),sum(v['fp'] for v in per.values()),sum(v['fn'] for v in per.values()))
    return a,per

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    detector=ProduceDetector(cfg)
    rows=[]
    started=time.perf_counter()
    for i,row in enumerate(manifest['sample'],1):
        if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()!=row['image_sha256']:raise ValueError('Input hash changed')
        if row.get('label_path') and hashlib.sha256(Path(row['label_path']).read_bytes()).hexdigest()!=row['label_sha256']:raise ValueError('Annotation hash changed')
        image=ImageOps.exif_transpose(Image.open(row['path'])).convert('RGB')
        boxes,ms=detector.predict(image)
        rows.append(dict(row,detections=boxes,latency_ms=round(ms,2),run_id=cfg['run_id']))
        if i%25==0:print(json.dumps({'run_id':cfg['run_id'],'done':i,'total':len(manifest['sample'])}),flush=True)
    aggregate,per=summarize(rows,manifest['class_names'])
    negatives=[r for r in rows if r['kind']=='synthetic_negative']
    result={'created_at':datetime.now(timezone.utc).isoformat(),'complete':True,'success':True,
        'model':cfg,'manifest':str(args.manifest),'split':manifest['split'],'trained_locally':False,
        'planned_images':len(rows),'attempted_images':len(rows),'evaluated_images':len(rows),
        'iou_threshold':.5,'confidence_threshold':.25,'aggregate':aggregate,'per_class':per,
        'negative_controls':{'images':len(negatives),'false_detections':sum(len(r['detections']) for r in negatives)},
        'wall_seconds':round(time.perf_counter()-started,2),'images':rows,
        'limitations':['Class-stratified sample; not population accuracy or mAP.','Public data pretraining overlap cannot be ruled out.','New classes outside these annotations are not validated by this benchmark.']}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'run_id':cfg['run_id'],'aggregate':aggregate,'samples':len(rows),'negative_controls':result['negative_controls']}),flush=True)
    detector.close()

if __name__=='__main__':main()
