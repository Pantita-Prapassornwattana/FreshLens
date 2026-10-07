"""Conservative confidence/routing selection using VALIDATION ONLY. No weight updates."""
import copy,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
from benchmark_expansion import summarize
from produce_detector import suppress
from tests.evaluate_pretrained import match,metrics

ART=Path(__file__).resolve().parent/'artifacts'
GRID=(.25,.35,.5,.65)

def score_class(rows,name,threshold):
    tp=fp=fn=0
    for row in rows:
        truth=[t for t in row['truth'] if t['class']==name]
        predictions=[b for b in row['detections'] if b['class']==name and b['confidence']>=threshold]
        hits=len(match(predictions,truth))
        tp+=hits;fp+=len(predictions)-hits;fn+=len(truth)-hits
    return metrics(tp,fp,fn)

if __name__=='__main__':
    candidates=[json.loads((ART/name).read_text(encoding='utf-8')) for name in (
        'expansion_baseline_validation.json','expansion_specialist_validation.json','expansion_v3_validation.json')]
    baseline=candidates[0]
    names=list(baseline['per_class'])
    selected={}
    selection_metrics={}
    for name in names:
        current=baseline['per_class'][name]
        winner=(0,.25,current)
        originally_supported=name in baseline['model']['members'][0]['class_map'].values()
        for index,candidate in enumerate(candidates):
            if name not in candidate['model']['members'][0]['class_map'].values():continue
            for threshold in GRID:
                result=score_class(candidate['images'],name,threshold)
                permitted=result['tp']>=(5 if originally_supported else 1) and (result['precision'] or 0)>=(.5 if originally_supported else .6)
                if permitted and (result['f1'] or 0)>(winner[2]['f1'] or 0)+.05:
                    winner=(index,threshold,result)
        selected[name]={'candidate':winner[0],'threshold':winner[1]}
        selection_metrics[name]={'baseline':current,'selected':winner[2]}
    members=[]
    for index,candidate in enumerate(candidates):
        member=copy.deepcopy(candidate['model']['members'][0])
        member['active_classes']=[n for n in names if selected[n]['candidate']==index and n in member['class_map'].values()]
        member['class_thresholds']={n:selected[n]['threshold'] for n in member['active_classes']}
        if member['active_classes']:members.append(member)
    lookups=[{r['path']:r for r in c['images']} for c in candidates]
    rows=[]
    for row in baseline['images']:
        boxes=[]
        for name,choice in selected.items():
            boxes.extend(b for b in lookups[choice['candidate']][row['path']]['detections'] if b['class']==name and b['confidence']>=choice['threshold'])
        rows.append(dict(row,detections=suppress(boxes)))
    aggregate,per=summarize(rows,names)
    cfg={'run_id':'produce-calibrated-v3','model':'Produce specialists · calibrated',
        'mode':'pretrained_ensemble','trained_locally':False,'members':members,'weights':members[0]['weights'],
        'classes':names,'evaluation_classes':names,'merge_iou':.45,'confidence_threshold':.25,
        'supported_classes':sorted({n for m in members for n in m['active_classes']}),
        'unsupported_classes':sorted(set(names)-{n for m in members for n in m['active_classes']}),
        'selection_method':'Validation only: F1 gain >5pp, >=5 true positives and precision >=.5; missing classes >=1 TP and precision >=.6; confidence grid .25/.35/.5/.65; no tile inference.',
        'validation_metrics':aggregate,'frozen_at':datetime.now(timezone.utc).isoformat(),
        'evaluation_artifact':'artifacts/calibrated_final_test.json','source':'https://huggingface.co/Piyu12/fruit-veg-yolo11m-detector'}
    for member in cfg['members']:member['weights_sha256']=hashlib.sha256((ART.parent/member['weights']).read_bytes()).hexdigest()
    (ART/'frozen_calibrated_core.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    (ART/'calibrated_selection.json').write_text(json.dumps({'config':cfg,'choices':selected,'selection_metrics':selection_metrics,'test_used_for_selection':False},ensure_ascii=False,indent=2),encoding='utf-8')
    result=dict(baseline,model=cfg,aggregate=aggregate,per_class=per,images=rows)
    (ART/'calibrated_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'validation':aggregate,'active_members':[{m['id']:m['class_thresholds']} for m in members]}),flush=True)
