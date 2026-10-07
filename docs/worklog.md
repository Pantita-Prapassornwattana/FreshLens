# บันทึกพัฒนาการ — 7 ตุลาคม 2026

| ขั้นตอน | สิ่งที่ทำ | หลักฐาน / สถานะ |
|---|---|---|
| 1 | สำรวจ workspace | โฟลเดอร์เริ่มต้นไม่มี source code หรือ Dataset |
| 2 | สำรวจ hardware | RTX 3050, VRAM 4096 MiB; Python alias ใช้ไม่ได้ จึงสร้าง .venv จาก bundled Python 3.12.14 |
| 3 | ตรวจ source Dataset | หน้า overview 47 คลาส / 41,985 ภาพ / CC BY 4.0 |
| 4 | ตรวจเวอร์ชัน | URL version 3 redirect ไป v8; 34008 train / 4619 valid / 3358 test; stretch 416x416 |
| 5 | ขอ download | HTTP ติด Cloudflare; เปิดผ่าน browser ได้ แต่ Export ต้อง Login จึงยังไม่ได้รับ ZIP |
| 6 | เตรียมเว็บ | HTML ภาษาไทย 3 tabs, upload, API, missing-model state, history, download report |
| 7 | เตรียม audit | ตรวจรูป/YOLO labels, duplicate pixel/source, freeze manifest, seed 42 |
| 8 | เตรียม experiment | YOLO11n vs YOLOv8n, nested subsets 10/30/100%, early stopping, validation selection, sealed test |
| 9 | รายงาน | รายงานทุกหัวข้อ; ไม่อ้าง mAP หรือ improvement ที่ยังไม่เกิดขึ้น |

แผนเดิม ณ ตอนเริ่มโครงการ (ยกเลิกส่วนเทรนตามคำสั่งล่าสุด ดูผลสำเร็จท้ายไฟล์): เข้าสู่ระบบเพื่อ Export; ดาวน์โหลด/แตก ZIP; audit และตรวจ distribution/near duplicates; smoke training; เทรนหลัก 6 runs; เปรียบเทียบ validation; final test; external labeled test; สรุปผลจริงเพิ่มในรายงาน

เหตุผลการออกแบบ: ใช้ nano เพราะ VRAM จำกัด; เริ่มใหม่แต่ละ subset เพื่อเปรียบเทียบ learning curve อย่างตรงไปตรงมา; ไม่ใช้ test เลือกโมเดล; ไม่เพิ่ม RNN/NLP/RL เพียงเพื่อให้มีชื่อเทคนิค

อัปเดต: ติดตั้ง torch 2.7.1+cu128 และ torchvision 0.22.1+cu128 สำเร็จ; บันทึก dependencies ใน requirements-lock.txt; ยังไม่มี Dataset และหน้า Roboflow ยังแสดง Login

อัปเดต: พบไฟล์ Combined Vegetables - Fruits.v8-2025-06-05-12-15am.yolov11 (2).zip ใน Downloads ตรวจ data.yaml ยืนยัน version 8 / 47 classes / 83982 ZIP entries ขนาด 1050186290 bytes และย้ายไป data/dataset-v8-yolov11.zip สำเร็จ เริ่มแตกไฟล์แล้ว ยังไม่มีผลเทรน

อัปเดต: แตก Dataset สำเร็จ; ดาวน์โหลด models/yolo11n.pt และ models/yolov8n.pt จาก Ultralytics สำเร็จ; PyTorch ตรวจพบ NVIDIA GeForce RTX 3050 Ti Laptop GPU และ CUDA available=True; เริ่ม prepare.py และตั้ง pipeline experiment.ps1 ให้รันหลัง audit สำเร็จ (ยังไม่อ้างผลเทรนจน artifacts/experiments.json ระบุ completed)

## สรุปการเปลี่ยนขอบเขตและผลสำเร็จ — 7 ตุลาคม 2026

คำสั่งล่าสุดให้แยก subagent หา Dataset/โมเดล ตรวจเว็บ และ **ไม่เทรน** จึงหยุด pipeline audit/training เดิม งานที่ระบุว่าต้องเทรนด้านบนเป็นแผนในอดีตและถูกยกเลิกแล้ว ไม่มี training run สำเร็จหรือปรับน้ำหนักในโครงการนี้

- แยกงาน 3 subagents: ค้นโมเดล/แหล่งข้อมูล, พัฒนาเว็บ, และทดสอบประเมินผล
- Dataset v8 ดาวน์โหลดและแตกครบ 41,985 ภาพ ใช้เป็นข้อมูลประเมินเท่านั้น; full audit เดิมหยุดก่อนเสร็จ
- เปรียบเทียบ pretrained 4 โมเดลด้วย validation 108 ภาพ เลือก YOLOv8m ของ Henning Heyen ด้วย F1 39.36% (YOLO-World small 4.77%, medium 7.28%, YOLOE small 6.40%)
- Freeze checkpoint/settings ก่อน final test 103 ภาพ: TP729 FP613 FN1298; Precision54.32% Recall35.96% F1 43.28%; สำเร็จ103/103 ไม่มี API errors
- โมเดลจับคู่ชื่อได้42/47คลาส แต่มี TPจริง39คลาสในชุดทดสอบ ไม่อ้างว่าแม่นทุกชนิด ไม่มีการปรับจากผล final test
- Unit/HTTP tests ผ่าน8ข้อ; ตรวจ malformed image, cross origin, เกิน12MB, schema/ภาพผลลัพธ์/ประวัติ และ PNG30MP ถูกปฏิเสธ413
- Browser QA ผ่าน: ตัวอย่างแอปเปิล2ผล, อัปโหลดไฟล์กล้วยตรวจได้1ผล, ภาพว่าง0ผล, หน้า metrics/ประวัติ/รายงานแสดงผลจริง
- บันทึกภาพ artifacts/web_apple.jpg, web_banana.jpg, web_evaluation.jpg; รายงาน docs/report.md; ผลรายภาพ/รายคลาส artifacts/pretrained_evaluation.json และ per_class_results.csv
- รันเว็บ localhost:8000 พร้อมใช้งาน; เปิดใหม่ด้วย run.ps1; epochs0 และโมเดลที่เทรนในโครงการ0

## ปรับปรุงเมื่อผู้ใช้แจ้งว่าตรวจผักบางชนิดไม่ได้

- แยกงาน subagents ตรวจโมเดล/manifest/เว็บ; ดาวน์โหลด Piyu12 YOLO11m35labels และ HenningHeyen v3 พร้อมSHA256 ไม่ปรับน้ำหนัก
- ทดสอบ validation241ภาพ/4047GT: เดิมF138.90%, specialist6.25%, v336.43%, crop32.40%, resolution96029.75%, classrouting41.61%, calibrated41.06%, supplement39.30%
- Classroutingที่validationดูดีได้test134ภาพF146.67%ต่ำกว่าเดิม48.17% จึงปฏิเสธ เก็บผลความล้มเหลวไว้ ไม่อ้างว่าแม่นขึ้น
- ทดลองการรักษาผลหลักและเติมกรอบจากโมเดลเสริม เลือก confidence .5/.5จากvalidation; confirmationใหม่120ภาพ F162.43%เทียบเดิม62.30%; ตรึงก่อนfinalชุดใหม่
- Finalpaired111ภาพ/2409GT/33observedclasses: TP925→963, FP445→510, FN1484→1446; Precision67.52→65.38%, Recall38.40→39.98%, F148.95→49.61% ดีขึ้นเล็กน้อย แลกกับfalse positivesเพิ่ม ไม่ตีความว่าแม่นทุกชนิด
- Negativecontrolในcoremodeltestเดิม0→รุ่นเสริม1falsepositive (whiteเป็นbroccoli) เก็บrawคะแนน unchanged แก้productionด้วยuniformcanvasguardช่วงRGBแต่ละchannel<=2; ทดสอบวัตถุเล็กบนพื้นขาวไม่ถูกกรอง
- Productionมี44ชื่อจากDataset47 เพิ่มbeet/cabbage +9ชนิดทดลองรวม53ชื่อ ไม่มีannotationวัด9ชนิดใหม่ ไม่รองรับbeans/egg/pattypansquash; ไม่มีtraining/epochs0
- ตรวจระบบผ่าน12tests และAPIหลัก; ภาพสาธิตกะหล่ำปลีตรวจได้84.6%แต่มีอะโวคาโดทายผิดเป็นกล้วย จึงยังมีข้อจำกัดชัดเจน ตัวอย่างไม่แทนคะแนนโดยรวม
- รายงานใหม่ docs/report.md, รายงานก่อนหน้า report_before_expansion.md, pairedmetrics expansion_comparison.json, rawfinal supplement_final_test.json, รูปกราฟ upgrade_comparison.png, frozen_production_upgrade.json
# ตรวจภาพสาธิตและเชื่อม GitHub (8 ตุลาคม 2026)

- แก้แท็บการเทรนที่ค้าง: เพิ่ม API สถานะ/CSV/รายงาน และ JavaScript โหลดสถานะพร้อม timeout และข้อความเมื่อโหลดไม่สำเร็จ
- อ่านผลจาก CSV ของ run ที่บันทึกจริงเท่านั้น; เมื่อไม่มีการฝึก แสดง 0 epoch และไม่มีข้อมูลกราฟ พร้อมกราฟเปรียบเทียบ pretrained เดิม/ระบบรวมที่วัดจริงบน 111 ภาพ
- ทดสอบ parser 3 ข้อ (ไม่มี run, metrics ขาด/NaN, ห้ามอ่านนอกโฟลเดอร์ runs) ผ่าน; ทดสอบ localhost endpoints และหน้าเว็บจริงแล้วไม่ค้างและไม่มี console error

- ตรวจภาพ `upgrade-beet` กับ annotation: มี beet, radish และ garlic จึงแก้ชื่อปุ่มที่เดิมระบุเพียง beet
- ตรวจผ่าน `/api/predict` พบผลผิด potato, onion, pear และ apple พร้อม beet หนึ่งกรอบ; ภาพบีทรูตเดี่ยวที่ตรวจเพิ่ม 6 ภาพยังมีผลผิด จึงไม่อ้างว่าแก้ความแม่นยำแล้ว
- แสดงปุ่มเป็นตัวอย่างข้อจำกัดที่ยังตรวจผิด ไม่ใช้ label ของปุ่มแทนผลจริงจากโมเดล
- เชื่อม repository `https://github.com/Pantita-Prapassornwattana/FreshLens` และจัด `.gitignore` ไม่เผยแพร่ Dataset, virtual environment, น้ำหนักโมเดล, cache และประวัติภาพของผู้ใช้
