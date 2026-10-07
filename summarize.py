"""Generate report appendix from actual artifacts, including incomplete experiments."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(name,default):
    p=ROOT/'artifacts'/name
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else default
def main():
    audit=read('dataset_audit.json',{})
    records=read('experiments.json',[])
    rows=['# ผลทดลองจากไฟล์จริง','',f'จำนวน runs สำเร็จ: {sum(r["status"]=="completed" for r in records)}',
          f'จำนวนภาพหลัง audit: {audit.get("counts",{})}',f'จำนวนไฟล์ที่ตัดออก: {len(audit.get("excluded",[]))}','',
          '| Run | Model | Train images | Epochs | Val mAP50–95 | Seconds | Status |',
          '|---|---|---:|---:|---:|---:|---|']
    for r in records:
        m=r.get('validation',{}).get('map50_95')
        rows.append(f'| {r["run_id"]} | {r["model"]} | {r["train_images"]} | {r.get("completed_epochs",0)} | {m if m is not None else "ยังไม่มี"} | {r.get("seconds",0)} | {r["status"]} |')
    completed=[r for r in records if r['status']=='completed']
    if len(completed)>1:
        baseline=completed[0]
        rows+=['','## ผลต่างเทียบ run แรก','',f'Baseline: {baseline["run_id"]}; เปรียบเทียบบน validation เดียวกัน ต้องอ่าน settings/epochs ประกอบ']
        for r in completed[1:]:
            delta=100*(r['validation']['map50_95']-baseline['validation']['map50_95'])
            rows.append(f'- {r["run_id"]}: ΔmAP50–95 {delta:+.2f} percentage points')
    test=read('final_test.json',None)
    rows+=['','## Final test','',json.dumps(test,ensure_ascii=False,indent=2) if test else 'ยังไม่ได้เปิด final test']
    auxiliary=read('features/metrics.json',None)
    if auxiliary:rows+=['','## การทดลองเสริม crop classification / PCA / K-means','',json.dumps(auxiliary,ensure_ascii=False,indent=2)]
    rows+=['','ผลจากภาพภายนอกที่เก็บใหม่: ยังไม่ได้ประเมิน; confidence ไม่รับประกันความถูกต้อง; ห้ามอ้างว่าการแบ่ง subset เพียงอย่างเดียวป้องกัน overfitting']
    (ROOT/'docs/results.md').write_text('\n'.join(rows),encoding='utf-8')
if __name__=='__main__':main()
