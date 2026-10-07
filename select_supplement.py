"""Select a global supplement threshold on validation; keep primary detections."""
import copy,hashlib,json
from itertools import product
from pathlib import Path
from datetime import datetime,timezone
from benchmark_expansion import summarize
from produce_detector import suppress
from tests.evaluate_pretrained import iou

ART=Path(__file__).resolve().parent/'artifacts'
if __name__=='__main__':
    baseline=json.loads((ART/'expansion_baseline_validation.json').read_text(encoding='utf-8'))
    supplements=[json.loads((ART/n).read_text(encoding='utf-8')) for n in ['expansion_v3_validation.json','expansion_specialist_validation.json']]
    lookups=[{r['path']:r for r in c['images']} for c in supplements]
    trials=[]
    for thresholds in product((.35,.5,.65),repeat=2):
        rows=[]
        for row in baseline['images']:
            boxes=copy.deepcopy(row['detections'])
            for lookup,threshold in zip(lookups,thresholds):
                existing=list(boxes)
                new=[b for b in lookup[row['path']]['detections'] if b['confidence']>=threshold and not any(iou(b['box'],old['box'])>.3 for old in existing)]
                boxes.extend(new)
            rows.append(dict(row,detections=suppress(boxes)))
        aggregate,per=summarize(rows,list(baseline['per_class']))
        trials.append({'thresholds':thresholds,'aggregate':aggregate,'per_class':per,'images':rows})
    best=max(trials,key=lambda t:t['aggregate']['f1'])
    members=[copy.deepcopy(baseline['model']['members'][0])]
    for candidate,threshold in zip(supplements,best['thresholds']):
        member=copy.deepcopy(candidate['model']['members'][0]);member['confidence_threshold']=threshold
        members.append(member)
    cfg={'run_id':'produce-supplement-v4','model':'Produce detector + high-confidence supplements',
        'mode':'pretrained_ensemble','members':members,'weights':members[0]['weights'],
        'classes':list(baseline['per_class']),'evaluation_classes':list(baseline['per_class']),
        'supported_classes':sorted({n for m in members for n in m.get('active_classes',m['class_map'].values())}),
        'supplement_only':True,'supplement_overlap':.3,'merge_iou':.45,'confidence_threshold':.25,
        'trained_locally':False,'frozen_at':datetime.now(timezone.utc).isoformat(),
        'selection_method':'Keep primary boxes; add secondary boxes with IoU <=.3 to prior boxes. Choose two global confidence thresholds from .35/.5/.65 using validation F1 only.',
        'validation_metrics':best['aggregate'],'evaluation_artifact':'artifacts/supplement_final_test.json'}
    cfg['unsupported_classes']=sorted(set(cfg['classes'])-set(cfg['supported_classes']))
    for member in members:member['weights_sha256']=hashlib.sha256((ART.parent/member['weights']).read_bytes()).hexdigest()
    (ART/'frozen_supplement_core.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    (ART/'supplement_selection.json').write_text(json.dumps({'trials':[{'thresholds':t['thresholds'],'aggregate':t['aggregate']} for t in trials],'chosen':best['thresholds'],'test_used_for_selection':False},indent=2),encoding='utf-8')
    result=dict(baseline,model=cfg,aggregate=best['aggregate'],per_class=best['per_class'],images=best['images'])
    (ART/'supplement_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'thresholds':best['thresholds'],'aggregate':best['aggregate']}))
