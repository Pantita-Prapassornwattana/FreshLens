"""Select class routing on validation only; preserve old frozen test evidence."""
import copy,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from benchmark_expansion import summarize
from produce_detector import suppress

ROOT=Path(__file__).resolve().parent
ART=ROOT/'artifacts'
if __name__=='__main__':
    candidates=[json.loads((ART/file).read_text(encoding='utf-8')) for file in (
        'expansion_baseline_validation.json','expansion_specialist_validation.json',
        'expansion_v3_validation.json','expansion_tiles_validation.json')]
    old,new=candidates[:2]
    candidates[3]['model']['members'][0]['id']='lvis63-tiles'
    names=list(old['per_class'])
    routing={}
    for name in names:
        best=max(candidates,key=lambda c:c['per_class'][name]['f1'] or 0)
        if (best['per_class'][name]['f1'] or 0)<=(old['per_class'][name]['f1'] or 0)+.03:best=old
        routing[name]=best['model']['members'][0]['id']
    members=[]
    for candidate in candidates:
        member=copy.deepcopy(candidate['model']['members'][0])
        member['active_classes']=[n for n in names if routing[n]==member['id'] and n in member['class_map'].values()]
        if member['active_classes']:members.append(member)
    by_path=[{r['path']:r for r in c['images']} for c in candidates]
    combined=[]
    for row in old['images']:
        boxes=[];latency=0
        for candidate,lookup in zip(candidates,by_path):
            original=lookup[row['path']]
            if original['image_sha256']!=row['image_sha256']:raise ValueError('Candidate inputs differ')
            identifier=candidate['model']['members'][0]['id']
            boxes.extend(b for b in original['detections'] if routing[b['class']]==identifier)
            if identifier in {m['id'] for m in members}:latency+=original['latency_ms']
        combined.append(dict(row,detections=suppress(boxes),latency_ms=latency))
    a,per=summarize(combined,names)
    cfg={'run_id':'produce-expanded-v2','model':'Produce specialists · ensemble',
         'mode':'pretrained_ensemble','members':members,'merge_iou':.45,
         'agnostic_merge':False,'trained_locally':False,'confidence_threshold':.25,
         'classes':names,'supported_classes':sorted({n for m in members for n in m['active_classes']}),
         'unsupported_classes':sorted(set(names)-{n for m in members for n in m['active_classes']}),
         'routing':routing,'selection_method':'per-class validation F1 improvement > 0.03; no test-based tuning',
         'validation_metrics':a,'source':'https://huggingface.co/Piyu12/fruit-veg-yolo11m-detector'}
    for member in cfg['members']:
        member['weights_sha256']=hashlib.sha256((ROOT/member['weights']).read_bytes()).hexdigest()
    result=dict(old,model=cfg,aggregate=a,per_class=per,images=combined,
                negative_controls={'images':4,'false_detections':sum(len(r['detections']) for r in combined if r['kind']=='synthetic_negative')})
    (ART/'expansion_ensemble_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    # Core-only configuration for the paired held-out comparison; extras lack annotations.
    (ART/'expansion_core_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    record={'created_at':datetime.now(timezone.utc).isoformat(),'routing':routing,
            'baseline':old['aggregate'],'specialist':new['aggregate'],'ensemble':a,
            'candidates':[{'run_id':c['model']['run_id'],'aggregate':c['aggregate']} for c in candidates],
            'test_used_for_selection':False,'weights_trained':False}
    (ART/'expansion_selection.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(record),flush=True)
