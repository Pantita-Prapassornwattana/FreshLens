"""Write final Thai report, CSV and plot from actual frozen pretrained evaluation."""
import csv
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ART=ROOT/'artifacts'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mpl'))
def read(name):return json.loads((ART/name).read_text(encoding='utf-8'))
def pct(value):return f'{value*100:.2f}%' if value is not None else 'N/A'

def main():
    if read('selected_model.json').get('members'):
        from generate_expansion_report import main as upgrade_main
        return upgrade_main()
    dataset=read('dataset_inventory.json');comparison=read('model_comparison.json')
    selected=read('selected_model.json');result=read('pretrained_evaluation.json')
    assert result['success'] and result['model']['weights_sha256']==selected['weights_sha256']
    a=result['aggregate'];detected=sum(m['tp']>0 for m in result['per_class'].values())
    latency=[r['latency_ms'] for r in result['images'] if r.get('latency_ms') is not None]
    with (ART/'per_class_results.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.writer(file);writer.writerow(['class','supported','tp','fp','fn','precision','recall','f1'])
        for name,m in result['per_class'].items():
            writer.writerow([name,name in selected['supported_classes'],*[m[k] for k in ('tp','fp','fn','precision','recall','f1')]])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4.5))
    for j,key in enumerate(('precision','recall','f1')):
        values=[100*c['metrics'][key] for c in comparison['candidates']]
        ax.bar([i+(j-1)*.24 for i in range(4)],values,width=.24,label=key)
    ax.set_xticks(range(4),['World S','World M','YOLOE S','Produce YOLOv8m'])
    ax.set(ylabel='Percent',title='Validation sample: confidence 0.25 / matching IoU 0.50',ylim=(0,60))
    ax.legend();fig.tight_layout();fig.savefig(ART/'model_comparison.png',dpi=150);plt.close(fig)
    table='\n'.join(f'| {c["model"]} | {c["images"]} | {pct(c["metrics"]["precision"])} | {pct(c["metrics"]["recall"])} | {pct(c["metrics"]["f1"])} | {c["classes_with_true_positives"]} |' for c in comparison['candidates'])
    delta=100*(comparison['candidates'][-1]['metrics']['f1']-comparison['candidates'][0]['metrics']['f1'])
    report=f'''# รายงาน Mini Project: FreshLens ระบบระบุชนิดผักผลไม้ด้วยโมเดลสำเร็จรูป

วันที่จัดทำ: 7 ตุลาคม 2026 · สถานะ: ใช้งานและทดสอบบน localhost แล้ว · ไม่เทรนเพิ่มตามคำสั่งล่าสุด

## 1. ที่มาและความสำคัญ

การแยกชนิดผักผลไม้ด้วยสายตาใช้เวลาและสับสนได้เมื่อรูปร่างใกล้เคียง ภาพมีหลายชนิดหรือวัตถุบังกัน โครงการสร้างเว็บภาษาไทยให้ผู้ใช้เลือกภาพและดูชื่อชนิด กรอบตำแหน่ง คะแนนโมเดล และเวลาประมวลผล โดยเก็บหลักฐานการประเมินที่ตรวจสอบย้อนหลังได้ ระบบไม่ได้ตรวจความสดหรือความปลอดภัยอาหาร

## 2. แนวคิดและแนวทางการแก้ปัญหา

ใช้ pretrained object detector แทนการเทรนใหม่ เปรียบเทียบโมเดลบน validation เดียวกันก่อนเลือก แล้วตรึงโมเดลและค่าทำนายเพื่อประเมิน test แยกต่างหาก เว็บประมวลผลบนเครื่องด้วย GPU ภาพผู้ใช้อัปโหลดไม่ส่งไปบริการ inference ภายนอก

โมเดลที่เลือกคือ **{selected['model']}** จับคู่ชื่อกับ Dataset ได้ **42 จาก 47 คลาส** ยังไม่รองรับ {', '.join(selected['unsupported_classes'])} การมีชื่อในรายการไม่ได้รับรองว่าแม่นยำทุกภาพ

## 3. วิธีการดำเนินงานและออกแบบ

### 3.1 ขั้นตอนและกระบวนการ

1. ดาวน์โหลด Roboflow v8 แบบ YOLOv11 ลง data/dataset-v8-yolov11.zip แล้วแตกลง data/raw ตรวจ data.yaml และจำนวนภาพจาก ZIP จริง
2. พบ train {dataset['counts']['train']:,} / validation {dataset['counts']['val']:,} / test {dataset['counts']['test']:,} ภาพ รวม 41,985 ภาพและ 47 คลาส; train ไม่ได้ใช้ปรับน้ำหนัก
3. หยุด pipeline เทรนและ audit เต็มชุดเดิมเมื่อผู้ใช้เปลี่ยนเป็นไม่เทรน การตรวจทุกไฟล์ยังไม่เสร็จ จึงไม่อ้างว่า audit ครบ Dataset ตัวอย่างที่ใช้ประเมินตรวจภาพ/annotation และเก็บ SHA256
4. หาและโหลด 4 pretrained candidates: YOLO-World small/medium, YOLOE small และ YOLOv8m ผักผลไม้ที่ผู้พัฒนาเทรนไว้แล้ว World/YOLOE ใช้ text embeddings เพื่อกำหนด vocabulary โดยไม่มี optimizer/training
5. ประเมิน validation 108 ภาพ (104 annotated + 4 synthetic negatives) เลือก 2 ภาพต่อคลาสและภาพหลายชนิดเพิ่ม ใช้ manifest/hash เดียวกันทุกโมเดล; confidence 0.25, NMS IoU 0.45, input 640
6. เลือกด้วย validation F1 จากการจับคู่ชื่อคลาสตรงกันและ IoU ≥ 0.50 แบบ one-to-one ตาม confidence บันทึก frozen_pretrained_model.json ก่อน final test
7. ประเมิน final test 103 ภาพ (99 annotated + 4 synthetic negatives) ครอบคลุม 47 คลาส เลี่ยง source stems ที่ตรง validation วิธีนี้อาจตัดเกินจริงเพราะชื่อทั่วไปชนกัน และยังไม่พิสูจน์ว่า scene/pretraining เป็นอิสระ
8. ทดสอบกดเว็บจริง การอัปโหลด ภาพว่าง bytes ไม่ใช่ภาพ origin ไม่อนุญาต และ payload เกินขนาด การประเมินไม่ปะปนในประวัติผู้ใช้

### 3.2 Workflow

```mermaid
flowchart LR
 A[Public dataset] --> B[Fixed validation sample]
 C[4 pretrained candidates] --> D[Validation inference]
 B --> D
 D --> E[Select by F1]
 E --> F[Freeze model/settings]
 F --> G[Separate test sample]
 F --> H[Local Python server]
 I[Upload or demo image] --> H
 H --> J[Class/box/confidence/latency]
 J --> K[History/report]
```

### 3.3 ระบบและเครื่องมือ

HTML/CSS/JavaScript frontend; Python HTTP server ที่ 127.0.0.1:8000; Python 3.12; Ultralytics 8.4.174; PyTorch 2.7.1+cu128; RTX 3050 Ti Laptop GPU 4 GB; Pillow; YAML; requests; matplotlib และ CSV/JSON ทุก dependency และโมเดลอยู่ในโฟลเดอร์งาน หน้าเว็บแสดงชื่อไทย ภาพ กรอบ คะแนน ประวัติ ผลรายชนิด และรายงาน

### 3.4 เทคนิค AI/ML ที่ใช้จริง

| เทคนิค | การใช้และขอบเขต |
|---|---|
| Regression | หัว detector ทำนายพิกัด/ขนาด bounding box ไม่ได้ทำนายราคา น้ำหนัก หรือความสด |
| Classification | จำแนกคลาสของวัตถุจาก feature ของ detector |
| Neural Network / Deep Learning / CNN | ใช้ pretrained YOLO backbone/head ไม่ฝึก neural network ใหม่ |
| NLP | ใช้ CLIP/MobileCLIP text embeddings ในการทดลอง World/YOLOE; detector ที่เลือกใช้คลาสคงที่ |
| Evaluation & Optimization | เลือก 4 candidates ด้วย validation ก่อน test; ไม่ใช้ test ปรับ threshold |
| Bayesian inference | ไม่ใช้; confidence ไม่ใช่ Bayesian posterior ที่ผ่าน calibration |
| Unsupervised Learning / Dimensionality Reduction | ไม่ได้รัน K-means/PCA; analyze_features.py เป็นแผนเดิมที่ยังไม่รัน ไม่อ้างผล |
| RNN / RL | ไม่ใช้ เพราะเป็นภาพนิ่ง ไม่มี temporal sequence หรือ environment/action/reward |

สไลด์ที่ผู้ใช้แนบยังอ่านผ่านเครื่องมือไม่ได้ จึงไม่รับรองความสอดคล้องกับข้อกำหนดเฉพาะทุกหน้า

## 4. ผลการดำเนินงานและการวิเคราะห์ผล

### 4.1 เปรียบเทียบ validation เดียวกัน

| Candidate | ภาพ | Precision | Recall | F1 | คลาสที่มี TP |
|---|---:|---:|---:|---:|---:|
{table}

โมเดลที่เลือกเพิ่ม validation F1 **{delta:.2f} percentage points** เทียบ World small เป็นผลการเลือกโมเดลสำเร็จรูป ไม่ใช่การดีขึ้นจากการเทรน กราฟอยู่ที่ artifacts/model_comparison.png

### 4.2 ผล final test

- สำเร็จ {result['evaluated_images']}/{result['planned_images']} ภาพ; API errors {len(result['failures'])}
- Ground-truth {a['tp']+a['fn']:,} วัตถุ; TP {a['tp']}, FP {a['fp']}, FN {a['fn']}
- Precision **{pct(a['precision'])}**, Recall **{pct(a['recall'])}**, F1 **{pct(a['f1'])}**
- มี TP อย่างน้อย 1 วัตถุใน **{detected} คลาส** จาก 47 คลาส Dataset (รองรับ 42 ตาม mapping)
- Synthetic negatives {result['negative_controls']['images']} ภาพ ตรวจพบผิด {result['negative_controls']['false_detections']} วัตถุ
- Mean inference latency {sum(latency)/len(latency):.2f} ms รวม warm-up ไม่รวม model load, decode, plot หรือ HTTP จึงไม่ใช่ end-to-end latency

ผลรายคลาสครบใน artifacts/per_class_results.csv และ pretrained_evaluation.json มี prediction/ground truth/IoU matches/image-label hash/latency ทุกภาพ ตัวเลขเป็น fixed-threshold precision/recall/F1 ไม่ใช่ mAP หรือเปอร์เซ็นต์ความถูกต้องของทุกภาพในชีวิตจริง รายละเอียดวิธีทดสอบเพิ่มใน docs/pretrained_validation.md

### 4.3 วิเคราะห์และข้อจำกัด

ระบบระบุชนิดและวาดกรอบจริงได้โดยไม่เทรน แต่ Recall ยังต่ำในภาพตลาด วัตถุเล็ก ภาพบังกัน และคลาสคล้ายกัน เป็นต้นแบบที่ตรวจข้อผิดพลาดย้อนหลังได้ ยังไม่รับรองความแม่นยำครบทุกชนิด ภาพสาธิตเลือกจากผลที่ตรวจจับถูกเพื่อสาธิต ไม่ใช่ตัวอย่างสุ่มหรือหลักฐานว่าแม่นทุกภาพ

โมเดลมี 63 labels จับคู่ synonyms ได้ 42 คลาส ตัด uppercase Strawberry/Tomato ที่ผู้พัฒนาระบุเป็น labels ผิดก่อนประเมิน; vegetable marrow จับคู่ zucchini/courgette มีความกำกวม ต้องอ่านผลรายคลาสประกอบ

Dataset สาธารณะอาจซ้ำกับ pretraining จึงยังไม่พิสูจน์ generalization ไปภาพถ่ายใหม่; ภาพถูก stretch 416×416 และ synthetic negatives ไม่แทนภาพธรรมชาติที่ไม่มีผักผลไม้ ควรเพิ่ม external annotated test, unknown handling และแยก scene/source ตาม provenance

### 4.4 Overfitting และทำซ้ำ

**โมเดลที่เทรนในโครงการ 0 ตัว / training epochs 0 / ภาพที่ใช้ปรับน้ำหนัก 0** ไม่มี learning curve หรือหลักฐานลด overfitting จาก fine-tuning การเลือกด้วย validation และเก็บ test แยกช่วยลดการปรับตาม test แต่ยังมี validation selection bias ไม่ใช้ผล test นี้ปรับต่อ Subset10/30/100 และ early stopping เป็นแผนเดิมที่หยุด ไม่ใช่ผลจริง

requirements-lock.txt เก็บ versions; frozen_pretrained_model.json เก็บ weights SHA256/settings; manifests เก็บ inputs/hashes; evaluator ไม่เทรนและปฏิเสธ run_id ที่เปลี่ยนกลางทดสอบ

## แหล่งอ้างอิง

- [Roboflow v8 / CC BY 4.0](https://universe.roboflow.com/yolo-jpkho/combined-vegetables-fruits/dataset/8)
- [ผู้พัฒนา pretrained Fruits-And-Vegetables Detector](https://github.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset)
- [Original weights](https://drive.google.com/drive/folders/1I4mtQK11C3p41pO9raR0trgPVj0eQ2yb)
- [YOLO-World](https://docs.ultralytics.com/models/yolo-world), [YOLOE](https://docs.ultralytics.com/models/yoloe)
- รายละเอียดที่มาและ license: docs/model_sources.md; ไม่คัดลอก publisher benchmarks มาแทน local results
'''
    (ROOT/'docs/report.md').write_text(report,encoding='utf-8')
    readme='''# FreshLens — ตรวจจับผักผลไม้บน localhost โดยไม่เทรน

เว็บ http://localhost:8000 · เลือกภาพเองหรือกดภาพตัวอย่าง เพื่อดูชื่อไทย กรอบ คะแนนโมเดล และเวลา

## เปิดใช้งาน

```powershell
./run.ps1
```

Dependencies อยู่ใน .venv; โมเดล pretrained อยู่ใน models; Dataset อยู่ใน data/raw และ ZIP ต้นฉบับ data/dataset-v8-yolov11.zip เมื่อติดตั้งใหม่ใช้ setup.ps1; venv อ้าง Python runtime ของเครื่อง จึงควรสร้างใหม่หากย้ายเครื่อง

ไม่ต้องรัน train.py หรือ experiment.ps1 Pipeline เทรนเดิมหยุดแล้วตามคำสั่ง ไม่มีการเทรนในโครงการ โมเดลที่ใช้ถูกฝึกโดยผู้เผยแพร่และนำมาทำ inference

## โมเดลและขอบเขต

ใช้ YOLOv8m ผักผลไม้ pretrained ของ Henning Heyen มี63labels จับคู่42คลาส Dataset ยังไม่รองรับ beans, beet, cabbage, egg, pattypan squash ชื่อคลาสไม่รับรองความแม่นยำทุกภาพ และ confidence ไม่ใช่ calibrated probability

เลือกจาก validation เทียบกับ YOLO-World small/medium และ YOLOE small ก่อน final test103ภาพ ได้ Precision54.32%,Recall35.96%,F1 43.28%;39คลาสมี TP ปัญหาหลักคือภาพแน่น วัตถุเล็ก และชนิดคล้ายกัน

## หลักฐานและรายงาน

- docs/report.md: รายงานภาษาไทย4หัวข้อพร้อมผลจริงและ workflow
- docs/pretrained_validation.md และ docs/model_sources.md: วิธีทดสอบ ที่มา ข้อจำกัด
- docs/worklog.md: บันทึกพัฒนาการ
- artifacts/model_comparison.json/.png: validation comparison4โมเดล
- artifacts/pretrained_evaluation.json และ per_class_results.csv: testรายภาพ/รายคลาส
- artifacts/*_manifest.json: inputs/ground truth/hashes
- artifacts/frozen_pretrained_model.json: checkpoint/settings/hash ก่อนtest
- artifacts/api_checks.json: HTTP contract tests
- artifacts/history.json และ predictions/: ประวัติผู้ใช้ ไม่เก็บภาพต้นฉบับ

## ทำซ้ำ

```powershell
./.venv/Scripts/python.exe -m unittest discover -s tests -v
./.venv/Scripts/python.exe tests/evaluate_pretrained.py --manifest artifacts/pretrained_evaluation_manifest.json --output artifacts/reproduction.json
./.venv/Scripts/python.exe generate_pretrained_report.py
```

Reproduction ใช้checkpoint/settingsเดิมตรวจการทำซ้ำ ไม่ควรใช้testนี้ปรับใหม่ ภาพสาธิตเลือกจากผลตรวจจับถูกและไม่แทนความแม่นยำโดยรวม ดู tests/README.md และดาวน์โหลดรายงานผ่าน /api/report
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    print('Report, README, CSV and comparison plot saved')
if __name__=='__main__':main()
