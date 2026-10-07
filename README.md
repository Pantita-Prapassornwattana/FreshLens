# FreshLens — ตรวจจับผักผลไม้ด้วย pretrained models

เปิด http://localhost:8000 หรือรัน `./run.ps1` ด้วยPowerShell

## เปิดโปรเจกต์จาก GitHub บน Windows

```powershell
git clone https://github.com/Pantita-Prapassornwattana/FreshLens.git
cd FreshLens
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\python.exe download_original_model.py
.\.venv\Scripts\python.exe download_original_model.py --file-id 1aaRDJWTYZKBEnt9wq719C9AKi8JskWDP --output models/fruit_vegetable_v3.pt
.\.venv\Scripts\python.exe download_expansion_model.py
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

เปิด `http://localhost:8000/` และเปิด PowerShell ค้างไว้ กด Ctrl+C เพื่อหยุด
รุ่นปัจจุบันใช้ CUDA GPU; เครื่องอื่นต้องมี PyTorch ที่รองรับ CUDA และ GPU ที่ใช้งานได้ก่อนตรวจจับภาพ
GitHub เก็บโค้ด ผลการประเมิน และเอกสาร แต่ไม่เก็บ `.venv`, Dataset และไฟล์น้ำหนักโมเดลขนาดใหญ่
Dataset สำหรับภาพตัวอย่างและทำซ้ำการประเมิน: [Roboflow Combined Vegetables & Fruits v8](https://universe.roboflow.com/yolo-jpkho/combined-vegetables-fruits/dataset/8) ดาวน์โหลด YOLO แล้วจัดไว้ใน `data/raw/` ให้มี `data.yaml`, `train/`, `valid/`, `test/`
การอัปโหลดภาพของผู้ใช้ทำงานได้เมื่อโหลดน้ำหนักครบ โดยไม่ต้องดาวน์โหลด Dataset ทั้งชุด

## ภาพตัวอย่างบีทรูต

ตัวอย่างเดิมเป็นภาพรวมบีทรูต หัวไชเท้า และกระเทียม ไม่ใช่บีทรูตเดี่ยว
ตรวจผ่าน API จริงแล้วพบกรอบผิดหลายคลาส จึงแก้ชื่อปุ่มให้บอกประเภทในภาพและระบุว่าเป็นตัวอย่างที่ยังตรวจผิด
เก็บตัวอย่างนี้ไว้แสดงข้อจำกัด ไม่ใช้ชื่อปุ่มบังคับผลทำนาย และไม่ได้เปลี่ยน threshold เพื่อให้ภาพนี้ดูสำเร็จ

รุ่นปัจจุบันproduce-supplement-v4ใช้โมเดลหลัก+โมเดลเสริม ไม่มีการเทรน ตัวเลือก44คลาสเดิมและ9ชนิดเพิ่มเติมทดลอง รวม53ชื่อ ยังไม่รองรับbeans,egg,pattypan squash

เทียบบน111ภาพเดียวกัน: Recall 38.40% → 39.98%, F1 48.95% → 49.61%, Precision 67.52% → 65.38% ผลดีขึ้นเล็กน้อยและตรวจผิดเพิ่ม ชนิดใหม่9ชนิดยังไม่มีคะแนนannotated

รายงานครบ4หัวข้อ: docs/report.md; pairedresults artifacts/expansion_comparison.json; rawresults supplement_final_test.json; perclass per_class_results.csv; provenance docs/model_expansion_sources.md; logs docs/worklog.md

โฟลเดอร์dataมีDataset41985ภาพกับZIPต้นฉบับ; modelsมีpretrainedweights; .venvมีdependenciesของเครื่องนี้; สร้างใหม่ด้วยsetup.ps1หากย้ายเครื่อง

ทดสอบระบบ: `./.venv/Scripts/python.exe -m unittest discover -s tests -v`
ทำซ้ำcorefinalโดยไม่เทรน: `./.venv/Scripts/python.exe benchmark_expansion.py --config artifacts/frozen_selected_upgrade.json --manifest artifacts/calibrated_test_manifest.json --output artifacts/reproduction_upgrade.json`
รายงานทำซ้ำ: `./.venv/Scripts/python.exe generate_expansion_report.py`

Testนี้ใช้ตรวจreproducibility ไม่ใช้ปรับโมเดลเพิ่มเติม No training: อย่ารันtrain.pyหรือexperiment.ps1
