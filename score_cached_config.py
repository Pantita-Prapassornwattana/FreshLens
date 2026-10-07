"""Compose predictions from independent full-model validation runs. Not a test evaluator."""
import argparse,copy,json
from pathlib import Path
from benchmark_expansion import summarize
from produce_detector import suppress
from tests.evaluate_pretrained import iou

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--prefix',required=True);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();cfg=json.loads(args.config.read_text(encoding='utf-8'))
    sources={}
    first=None
    for name in ('baseline','v3','specialist'):
        result=json.loads(Path(f'artifacts/{args.prefix}_{name}.json').read_text(encoding='utf-8'))
        if first is None:first=result
        sources[result['model']['members'][0]['id']]={r['path']:r for r in result['images']}
    rows=[]
    for row in first['images']:
        boxes=[];latency=0
        for index,member in enumerate(cfg['members']):
            source=sources[member['id']][row['path']]
            if source['image_sha256']!=row['image_sha256']:raise ValueError('Candidate input mismatch')
            allowed=set(member.get('active_classes',member['class_map'].values()))
            threshold=member.get('confidence_threshold',.25)
            new=[b for b in source['detections'] if b['class'] in allowed and b['confidence']>=member.get('class_thresholds',{}).get(b['class'],threshold)]
            if cfg.get('supplement_only') and index>0:
                new=[b for b in new if not any(iou(b['box'],old['box'])>cfg.get('supplement_overlap',.3) for old in boxes)]
            boxes.extend(new);latency+=source['latency_ms']
        rows.append(dict(row,detections=suppress(boxes),latency_ms=latency))
    aggregate,per=summarize(rows,cfg['evaluation_classes'])
    output=dict(first,model=cfg,aggregate=aggregate,per_class=per,images=rows,
                composition='Cached independent full-model validation outputs, class-specific NMS; final held-out test uses actual engine.',
                negative_controls={'images':4,'false_detections':sum(len(r['detections']) for r in rows if r['kind']=='synthetic_negative')})
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'run_id':cfg['run_id'],'aggregate':aggregate}))
