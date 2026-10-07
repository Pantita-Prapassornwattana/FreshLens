# การตรวจสอบโมเดลสำเร็จรูปโดยไม่เทรน

`evaluate_pretrained.py` ส่งไฟล์ภาพจริงไปที่ `/api/predict` ของเว็บ localhost แล้วเทียบชื่อคลาสและกรอบกับ annotation ของ Roboflow ไม่ได้เรียก training หรือโหลด PyTorch ภายในสคริปต์นี้

## วิธีวัด

- เลือกภาพ validation แบบกำหนด seed 47: เป้าหมาย 2 ภาพต่อคลาส และภาพหลายคลาสเพิ่ม 10 ภาพ
- เก็บรายการภาพ, SHA256 ภาพและ annotation, ground truth, prediction, confidence, latency และคู่กรอบที่ match ใน JSON
- ใช้ confidence ≥ 0.25 และ IoU ≥ 0.50; match ต้องเป็นคลาสเดียวกัน และแต่ละ prediction/ground truth จับคู่ได้หนึ่งครั้ง
- รายงาน TP, FP, FN, precision, recall, F1 รวมและรายคลาส ค่า `null` หมายถึงไม่มี denominator
- ไม่ใช้ชื่อ “mAP” เพราะ API คืนเฉพาะ prediction ที่ผ่าน threshold จึงไม่ได้คำนวณ AP ตลอดช่วง confidence
- มี negative controls สังเคราะห์ 4 ภาพ: สีขาว สีดำ checkerboard และรูปทรงเรขาคณิต ภาพเหล่านี้ไม่ใช่ตัวแทนสิ่งของทั่วไปในโลกจริง
- การส่งภาพเพื่อประเมินใช้ header `X-Evaluation: true` จึงไม่ทำให้ประวัติใช้งานของผู้ใช้เต็มไปด้วยรายการทดสอบ

ภาพ validation ที่เลือกจริงมี 104 ภาพ พร้อม negative controls 4 ภาพ; annotation รวม 1,899 วัตถุ ครบทั้ง 47 คลาส มีภาพหลายคลาส 68 ภาพ บางภาพตลาดมีวัตถุขนาดเล็กจำนวนมาก จึงควรอ่าน recall และผลรายคลาสประกอบ ห้ามสรุปว่าตรวจได้ครบ 47 ชนิดจากการกำหนด vocabulary อย่างเดียว

## คำสั่งทำซ้ำ

รันจากโฟลเดอร์โครงการ เมื่อเว็บและโมเดลที่เลือกพร้อมแล้ว:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_pretrained_evaluation -v
.\.venv\Scripts\python.exe tests/evaluate_pretrained.py --manifest artifacts/pretrained_validation_manifest.json --output artifacts/candidate_validation.json
```

ควรใช้ manifest เดียวกันเมื่อเปรียบเทียบโมเดล เมื่อเลือกโมเดลและตรึงค่าทั้งหมดแล้วจึงประเมิน test:

```powershell
.\.venv\Scripts\python.exe tests/evaluate_pretrained.py --manifest artifacts/pretrained_evaluation_manifest.json --output artifacts/pretrained_evaluation.json
```

หลังมีผลของทุก candidate และ final test แล้ว สรุปคะแนนโดยไม่รัน inference เพิ่ม และตรวจ HTTP contract ด้วยภาพสังเคราะห์:

```powershell
.\.venv\Scripts\python.exe -m tests.summarize_pretrained
.\.venv\Scripts\python.exe -m unittest tests.test_http_api -v
```

manifest test ที่เตรียมไว้มี 99 ภาพ annotation และ negative controls 4 ภาพ รวม 103 ภาพ มี 2,027 วัตถุ ครบ 47 คลาส และภาพหลายคลาส 76 ภาพ ตัดกลุ่มชื่อไฟล์ต้นทางที่เคยใช้ใน validation ออกก่อนสุ่ม; ชื่อทั่วไปของไฟล์ต้นทางอาจซ้ำกันทั้งที่เป็นภาพต่างกัน จึงเป็นการตัดออกแบบอนุรักษ์นิยม ไม่รับรองความเป็นอิสระของฉากหรือข้อมูล pretraining ที่ไม่เปิดเผย

## ข้อจำกัดของผล

ผลจากตัวอย่าง stratified ขนาดเล็กไม่ใช่ความแม่นยำของ dataset ทั้งหมดหรือการใช้งานจริง โมเดล pretrained อาจเคยเห็นภาพสาธารณะเหล่านี้แล้ว ข้อมูล annotation อาจมีชื่อคลาสที่คลุมเครือหรือวัตถุไม่ได้ติดป้ายไว้ครบ และการเลือกโมเดลจาก validation ต้องไม่เปลี่ยนต่อหลังเห็นคะแนน test ควรเก็บภาพภายนอกที่ถ่ายเองเพื่อประเมินเพิ่มเติมในอนาคต

Unit tests ตรวจการจับคู่กรอบและสูตร metrics ในกรณี prediction ซ้ำ, คลาสผิด, กรอบไม่ทับกัน และ denominator เป็นศูนย์ ไม่ตั้งคะแนนที่โมเดลต้องได้ไว้ล่วงหน้า
