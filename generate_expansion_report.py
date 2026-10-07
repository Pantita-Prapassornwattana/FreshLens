"""Build Thai report from measured upgrade trials and frozen paired test results."""
import csv,json,os,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ART=ROOT/'artifacts'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mpl'))
def read(n):return json.loads((ART/n).read_text(encoding='utf-8'))
def pct(v):return 'ไม่มีตัวหาร' if v is None else f'{100*v:.2f}%'

def main():
    cfg=read('selected_model.json');result=read('supplement_final_test.json');comparison=read('expansion_comparison.json')
    assert cfg['run_id']==result['model']['run_id']
    old=comparison['baseline'];new=comparison['upgraded']
    files=['expansion_baseline_validation.json','expansion_specialist_validation.json','expansion_v3_validation.json','expansion_tiles_validation.json','resolution960_validation.json','expansion_ensemble_validation.json','calibrated_validation.json','supplement_validation.json']
    labels=['โมเดลเดิม 640','YOLO11m 35 labels','ผู้สร้างเดิม v3','เดิม + crop tiles','เดิม 960','เลือกโมเดลรายคลาส','เลือกโมเดล + threshold รายคลาส','เดิม + โมเดลเสริม confidence สูง']
    reports=[read(n) for n in files]
    table='\n'.join(f'| {name} | {pct(r["aggregate"]["precision"])} | {pct(r["aggregate"]["recall"])} | {pct(r["aggregate"]["f1"])} |' for name,r in zip(labels,reports))
    missing=[n for n,m in result['per_class'].items() if m['tp']+m['fn']==0]
    median=statistics.median(r['latency_ms'] for r in result['images'])
    with (ART/'per_class_results.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.writer(file);writer.writerow(['class','supported','observed_in_final_test','tp','fp','fn','precision','recall','f1'])
        for n,m in result['per_class'].items():writer.writerow([n,n in cfg['core_supported_classes'],m['tp']+m['fn']>0,*[m[k] for k in ('tp','fp','fn','precision','recall','f1')]])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for i,key in enumerate(('precision','recall','f1')):ax.bar([i-.18,i+.18],[100*old[key],100*new[key]],width=.36,color=['#8aa275','#245441'])
    ax.set_xticks(range(3),['Precision','Recall','F1']);ax.set(ylim=(0,100),ylabel='Percent',title='Paired fresh test: 111 images / 2,409 annotated objects')
    ax.legend([plt.Rectangle((0,0),1,1,color='#8aa275'),plt.Rectangle((0,0),1,1,color='#245441')],['Original','Supplemented'])
    fig.tight_layout();fig.savefig(ART/'upgrade_comparison.png',dpi=160);plt.close(fig)
    report=f'''# รายงาน Mini Project: FreshLens ระบบระบุผักผลไม้บน localhost

อัปเดต 7 ตุลาคม 2026 · รุ่น {cfg['run_id']} · ไม่เทรนเพิ่มตามคำสั่งผู้ใช้

## 1. ที่มาและความสำคัญ

ระบบเดิมรู้จักชื่อที่จับคู่ได้42คลาส แต่ยังพลาดผักบางชนิดและวัตถุเล็กในภาพแน่น โมเดลที่มีชื่อคลาสอยู่ไม่ได้รับรองว่าจะตรวจได้ถูกทุกภาพ การเพิ่ม Dataset อย่างเดียวไม่ปรับความสามารถของโมเดลหากไม่มีการเทรน โครงการนี้ปรับใช้โมเดล pretrained เพิ่มและวัดผลจริงโดยไม่ปรับน้ำหนัก

## 2. แนวคิดและแนวทางการแก้ปัญหา

รักษาผลโมเดลหลัก แล้วใช้โมเดลเสริมค้นหาวัตถุที่ยังไม่พบ โดยรับเฉพาะคะแนนผ่านเกณฑ์และตำแหน่งที่ IoU กับกรอบก่อนหน้าไม่เกิน0.3 วิธีนี้เพิ่ม Recall แต่เสี่ยง false positives จึงเปรียบเทียบกับรุ่นเดิมบนภาพเดียวกันเสมอ ทดลองการแทนโมเดลตามคลาส การตัดภาพ และความละเอียดเพิ่มด้วย แต่เก็บผลที่ไม่ดีไว้ในรายงานและไม่ใช้เป็นค่าเริ่มต้น

## 3. วิธีการดำเนินงานและออกแบบ

### 3.1 กระบวนการ

1. แยก subagents ตรวจที่มาโมเดล จัดชุดข้อมูล และปรับหน้าเว็บ
2. ตรวจผลรายคลาสเดิม ดาวน์โหลด pretrained YOLO11m 35 labels และผู้สร้างเดิมv3; เก็บ revision/ขนาด/SHA256
3. ตรึง validation241ภาพ (237 annotated+4 controls),4047วัตถุ,47คลาส; ใช้เลือกวิธี/threshold
4. ทดลองหลายวิธี; trialแรกที่เลือกแทนรายคลาสมี F1 validationสูงขึ้น แต่ test134ภาพได้46.67% เทียบเดิม48.17% จึงปฏิเสธ ไม่อ้างว่าเป็นผลดี
5. เลือกแนวทางเสริมและ confidence .5/.5 จาก validationเท่านั้น; ใช้ confirmation validationใหม่120ภาพ,2522GT,39คลาส เลือกระหว่าง configurationsที่ตรึงแล้ว ไม่มีการปรับน้ำหนัก
6. ตรึงรุ่นที่เลือก แล้วประเมิน final testใหม่111ภาพ (107annotated+4controls),2409GT,33คลาส แบบpairedกับโมเดลเดิม ไม่มีการปรับ threshold/routing จากผล final testนี้
7. เปิดเว็บ อัปโหลดภาพจริง ตรวจ API/ภาพว่าง/ประวัติ/รายงาน และบันทึกหลักฐาน

### 3.2 Workflow

```mermaid
flowchart LR
 A[ผู้ใช้เลือกภาพ] --> B[ตรวจชนิดไฟล์ ขนาด และปรับ EXIF]
 B --> C[โมเดลหลักบนเครื่อง]
 C --> D[โมเดลเสริมตรวจบริเวณที่ยังไม่พบ]
 D --> E[กรองคะแนนและรวมกรอบ]
 E --> F[ชื่อไทย กรอบ คะแนน เวลา]
 F --> G[บันทึกประวัติและรุ่นโมเดล]
```

### 3.3 องค์ประกอบและเทคโนโลยี

เว็บภาษาไทย HTML/CSS/JavaScript; Python ThreadingHTTPServer bind127.0.0.1:8000; Ultralytics/PyTorch/Pillow; GPU RTX3050Ti Laptop4GB การ inference อยู่ในเครื่อง ไม่ส่งภาพไปบริการโมเดลภายนอก ระบบรับ JPG/PNG/WebPไม่เกิน12MB/25MP และเก็บประวัติเฉพาะผล ไม่เก็บภาพต้นฉบับ

produce_detector.py โหลด pretrained3checkpoints ใช้น้ำหนักเดิมทั้งหมด โมเดลหลักเกณฑ์.25; โมเดลเสริม.5; ชนิดเพิ่มเติมทดลอง.65; class-aware NMS IoU.45 และ suppress supplementary regionsเมื่อทับกรอบเดิม IoU>.3 ไม่มีการใช้ crop/960 ในรุ่นที่ส่งมอบเพราะผล validationไม่ดี

### 3.4 Dataset และขอบเขตชื่อ

Roboflow Combined Vegetables Fruits v8 / CC BY4.0 มี41985ภาพ:train34008/valid4619/test3358/47คลาส เก็บZIPและไฟล์แตกในdata ไม่ใช้trainเพื่อปรับน้ำหนัก รูปถูกstretch416×416; auditเดิมไม่เสร็จ จึงไม่อ้างว่าตรวจทั้ง41985ภาพแล้ว

จากเดิม42ชื่อ เพิ่มbeetและcabbageเป็น44ชื่อใน47คลาสเดิม ยังไม่รองรับbeans,egg,pattypan squash รวม9ชื่อใหม่จากโมเดลเสริมเป็น53ตัวเลือก: {', '.join(cfg['additional_classes'])} ชื่อใหม่9ชนิดเป็น**ทดลอง** ยังไม่มี annotationในDatasetนี้ จึงไม่มีคะแนนความแม่นยำรายชนิด และไม่จัดGreen_Orangeเป็นชนิดพืชใหม่

### 3.5 เทคนิค AI/ML

ใช้ CNN / Neural Network & Deep Learning เพื่อดึงคุณลักษณะภาพ; Classificationเพื่อชื่อชนิด; Bounding-box Regressionเพื่อตำแหน่ง; ensembleเพื่อเสริมวัตถุที่พลาด; evaluation/optimizationเพื่อเลือกthresholdจากvalidation ไม่ใช้Bayesian inference/RNN/RL/PCAหรือUnsupervised learningในรุ่นนี้ NLP/text promptingใช้ในYOLO-World/YOLOEรอบก่อนเท่านั้น Confidenceเป็นscoreไม่ใช่calibrated probability

## 4. ผลการดำเนินงานและการวิเคราะห์ผล

### 4.1 เปรียบเทียบ development validation241ภาพเดียวกัน

| วิธี | Precision | Recall | F1 |
|---|---:|---:|---:|
{table}

โมเดล35labelsเสียเปรียบในการวัด47คลาสรวม แต่ผลยังบอกว่าแทนโมเดลเดิมทั้งหมดไม่ได้ การเลือกผลดีที่สุดรายคลาสเสี่ยงselection biasแม้ไม่มีtraining overfit Trial test134ภาพที่ไม่ดีถูกเก็บในexpansion_final_test.json

Confirmation validation120ภาพ: เดิมF162.30%, calibrated62.20%, supplement62.43%; supplementมีRecall57.22% เทียบเดิม55.43% คะแนนF1เพิ่มเพียงเล็กน้อย ไม่ควรกล่าวว่าดีขึ้นมาก

### 4.2 Final paired test111ภาพเดียวกัน

| ตัวชี้วัด | เดิม | รุ่นเสริม |
|---|---:|---:|
| TP | {old['tp']} | {new['tp']} |
| FP | {old['fp']} | {new['fp']} |
| FN | {old['fn']} | {new['fn']} |
| Precision | {pct(old['precision'])} | {pct(new['precision'])} |
| Recall | {pct(old['recall'])} | {pct(new['recall'])} |
| F1 | {pct(old['f1'])} | {pct(new['f1'])} |

ตรวจพบถูกเพิ่ม38วัตถุ พลาดลด38 แต่ตรวจผิดเพิ่ม65 จึงเพิ่มRecall{100*(new['recall']-old['recall']):.2f}จุดเปอร์เซ็นต์และF1{100*(new['f1']-old['f1']):.2f}จุดเปอร์เซ็นต์ แลกกับPrecisionลด{100*(old['precision']-new['precision']):.2f}จุดเปอร์เซ็นต์ สรุปว่าจับวัตถุเพิ่มขึ้นและF1ดีขึ้นเล็กน้อย **ยังไม่ยืนยันว่าแม่นขึ้นทุกชนิด**

Negative controls4ภาพ: เดิมตรวจผิด0 รุ่นเสริมตรวจผิด{result['negative_controls']['false_detections']}ครั้ง จึงยังมีfalse positiveบนภาพไม่มีอาหาร รุ่นเสริมสำเร็จ111/111ภาพ; median inference{median:.1f}ms (ไม่รวมload model/HTTP/render;เครื่องอื่นต่างได้)

คะแนนตารางนี้เป็นผลโมเดลจากcoreconfigที่ตรึงก่อนfinal test เก็บไว้โดยไม่แก้ไข หลังทดสอบAPIพบโมเดลเสริมทายภาพสีขาวเป็นbroccoli จึงเพิ่มproduction input guardสำหรับภาพที่ช่วงค่าสีทุกchannelไม่เกิน2 ซึ่งไม่มีขอบวัตถุที่มองเห็น ไม่เปลี่ยนน้ำหนัก/threshold/routing ไม่หักFPย้อนหลังเพื่อแต่งคะแนน ทดสอบทั้งภาพสีพื้นและภาพมีวัตถุเล็กบนพื้นขาวแล้ว

จับคู่classตรงกัน + IoU≥.5 แบบone-to-one; ไม่ใช่mAP/accuracyของทุกภาพ FinalมีGT33คลาส ไม่มีGTสำหรับ: {', '.join(missing)} ค่าไม่มีตัวหารต้องแสดงN/A ไม่สรุปเป็นความสำเร็จของคลาสนั้น ภาพsampleมีการเลือกตามคลาส ไม่ใช่random-populationestimate และห้ามเทียบคะแนนชุด103/134/111ต่างกันเป็นbefore/after

### 4.3 การป้องกันการใช้ข้อมูลซ้ำและการตรวจระบบ

Manifestเก็บภาพ/annotation hashes ตัดsource groupsที่ใช้ก่อนหน้าและแยกconfirmation/final test ประวัติของชุดที่ปฏิเสธเก็บไว้ source stemsชื่อทั่วไปทำให้ตัดภาพจำนวนมากจนขาดบางคลาส การใช้publicdataยังอาจทับpretraining และชื่อไฟล์ไม่พิสูจน์ความเป็นอิสระของฉาก controls4patternsเดิมไม่ใช่หลักฐานnegativeที่เป็นภาพธรรมชาติ

**จำนวนโมเดลที่เทรนในโครงการ0 / epochs0 / ภาพที่ใช้ปรับน้ำหนัก0** ไม่มีlearningcurveหรือหลักฐานลดoverfittingจากtraining การแยกข้อมูลลดtest-driven selection แต่validation optimizationยังมีselectionbias;ต้องใช้externalphotosที่annotatedจริงเพื่อยืนยัน หากต้องการแม่นขึ้นมากหรือรองรับผักไทยที่ไม่มีในlabels ต้องเก็บภาพหลายสภาพและfine-tuneในงานถัดไป

ตรวจระบบผ่าน12tests: API malformed400/crossorigin403/เกิน12MB413/schema/history/ภาพผลลัพธ์, boxsuppression, uniform-image guard, evaluationmath และauditfixture; หลักฐานรอบล่าสุดอยู่artifacts/api_checks.json และupgrade_browser_checks.json ผลรายภาพsupplement_final_test.json ผลรายคลาสper_class_results.csv

### 4.4 หลักฐานและการทำซ้ำ

เปิดเว็บใหม่ด้วยrun.ps1; ไม่รันtrain.py/experiment.ps1; requirements-lock.txtเก็บversions; frozen_selected_upgrade.jsonตรึงcoreก่อนfinal; frozen_production_upgrade.jsonเพิ่ม9ชื่อทดลองโดยไม่เปลี่ยนcorethresholds; upgrade_baseline_test.jsonเก็บpairedbaseline; selected_model_v1.jsonและreport_before_expansion.mdเก็บรุ่นก่อน

แหล่งข้อมูล: [Roboflow v8](https://universe.roboflow.com/yolo-jpkho/combined-vegetables-fruits/dataset/8), [ผู้สร้างโมเดล63คลาส](https://github.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset), [Piyu12 YOLO11m](https://huggingface.co/Piyu12/fruit-veg-yolo11m-detector) รายละเอียดlicense/provenanceในdocs/model_expansion_sources.md ใช้คะแนนlocalจริง ไม่คัดลอกpublisherbenchmarkมาอ้างเป็นผลโครงการ
'''
    (ROOT/'docs/report.md').write_text(report,encoding='utf-8')
    (ROOT/'README.md').write_text(f'''# FreshLens — ตรวจจับผักผลไม้ด้วย pretrained models

เปิด http://localhost:8000 หรือรัน `./run.ps1` ด้วยPowerShell

รุ่นปัจจุบัน{cfg['run_id']}ใช้โมเดลหลัก+โมเดลเสริม ไม่มีการเทรน ตัวเลือก44คลาสเดิมและ9ชนิดเพิ่มเติมทดลอง รวม53ชื่อ ยังไม่รองรับbeans,egg,pattypan squash

เทียบบน111ภาพเดียวกัน: Recall {pct(old['recall'])} → {pct(new['recall'])}, F1 {pct(old['f1'])} → {pct(new['f1'])}, Precision {pct(old['precision'])} → {pct(new['precision'])} ผลดีขึ้นเล็กน้อยและตรวจผิดเพิ่ม ชนิดใหม่9ชนิดยังไม่มีคะแนนannotated

รายงานครบ4หัวข้อ: docs/report.md; pairedresults artifacts/expansion_comparison.json; rawresults supplement_final_test.json; perclass per_class_results.csv; provenance docs/model_expansion_sources.md; logs docs/worklog.md

โฟลเดอร์dataมีDataset41985ภาพกับZIPต้นฉบับ; modelsมีpretrainedweights; .venvมีdependenciesของเครื่องนี้; สร้างใหม่ด้วยsetup.ps1หากย้ายเครื่อง

ทดสอบระบบ: `./.venv/Scripts/python.exe -m unittest discover -s tests -v`
ทำซ้ำcorefinalโดยไม่เทรน: `./.venv/Scripts/python.exe benchmark_expansion.py --config artifacts/frozen_selected_upgrade.json --manifest artifacts/calibrated_test_manifest.json --output artifacts/reproduction_upgrade.json`
รายงานทำซ้ำ: `./.venv/Scripts/python.exe generate_expansion_report.py`

Testนี้ใช้ตรวจreproducibility ไม่ใช้ปรับโมเดลเพิ่มเติม No training: อย่ารันtrain.pyหรือexperiment.ps1
''',encoding='utf-8')
    print('Current Thai report, paired plot, CSV and README saved')

if __name__=='__main__':main()
