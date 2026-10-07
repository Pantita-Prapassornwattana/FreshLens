"""Select the best fixed-threshold pretrained candidate on validation only."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ART=ROOT/'artifacts'
configs={'world_small_validation.json':'world_config.json',
         'world_medium_validation.json':'world_medium_config.json',
         'yoloe_small_validation.json':'yoloe_config.json',
         'domain_validation.json':'domain_config.json'}
rows=[]
for report,config in configs.items():
    r=json.loads((ART/report).read_text(encoding='utf-8'))
    c=json.loads((ART/config).read_text(encoding='utf-8'))
    assert r['success'] and r['split']=='valid'
    rows.append({'run_id':r['model']['run_id'],'model':r['model']['model'],'config':config,
                 'images':r['evaluated_images'],'metrics':r['aggregate'],
                 'classes_with_true_positives':sum(v['tp']>0 for v in r['per_class'].values()),
                 'weights_sha256':c['weights_sha256'],'report':report})
winner=max(rows,key=lambda r:r['metrics']['f1'])
selection=json.loads((ART/winner['config']).read_text(encoding='utf-8'))
selection['validation']=winner['metrics']
selection['selection_method']='Highest F1 on same108-image validation sample, confidence0.25 / matchingIoU0.50; no training'
for name in ('selected_model.json','frozen_pretrained_model.json'):
    (ART/name).write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf-8')
(ART/'model_comparison.json').write_text(json.dumps({'selection_uses':'valid_only','candidates':rows,'selected':winner['run_id']},ensure_ascii=False,indent=2),encoding='utf-8')
print('Selected:',winner['run_id'])
