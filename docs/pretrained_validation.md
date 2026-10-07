# ผลการตรวจสอบโมเดลสำเร็จรูปสำหรับจำแนกและตรวจจับผักผลไม้

วันที่ดำเนินการ: 7 ตุลาคม 2569 (Asia/Bangkok)

## เป้าหมายและขอบเขต

ปรับระบบให้รับภาพจริงผ่านเว็บ localhost แล้วคืนชื่อชนิด ความเชื่อมั่น และกรอบตำแหน่ง โดยใช้โมเดลที่มีน้ำหนักสำเร็จรูป เปรียบเทียบ 4 โมเดล และไม่เทรนหรือ fine tune เพิ่มในขั้นตอนนี้ จำนวน training runs และ epochs ของขั้นตอนนี้จึงเป็น 0 ผลต่างระหว่างโมเดลเป็นผลจากการเลือกโมเดล pretrained คนละตัว ไม่ใช่ผลจากการเทรนให้ดีขึ้น

## Dataset และการแบ่งข้อมูล

ใช้ [Combined Vegetables Fruits v8 ของ Roboflow](https://universe.roboflow.com/yolo-jpkho/combined-vegetables-fruits/dataset/8) ซึ่งดาวน์โหลดและแตกไว้ใน `data/raw` มีชื่อคลาสใน `data.yaml` 47 คลาส รวม egg และ almond ตาม taxonomy ของแหล่งข้อมูล ตรวจพบ validation 4,619 ภาพ และ test 3,358 ภาพพร้อมไฟล์ annotation; test ไม่มีไฟล์ label ว่าง จึงเพิ่ม negative controls สังเคราะห์แยกต่างหาก

เลือก validation ด้วย seed 47 เป้าหมาย 2 ภาพต่อคลาสและภาพหลายคลาสเพิ่ม 10 ภาพ ได้ภาพ annotation 104 ภาพ พร้อม negative controls 4 ภาพ รวม 108 ภาพ มี annotation 1,899 วัตถุ ครบ 47 คลาส และภาพหลายคลาส 68 ภาพ ทุกโมเดลใช้ manifest เดียวกันและตรวจ SHA256 ของภาพและ annotation ก่อนใช้ซ้ำ

หลังเลือกโมเดลจาก validation จึงใช้ test อีกชุด โดยตัดกลุ่มชื่อไฟล์ต้นทางที่เคยใช้ใน validation ก่อนเลือก ได้ภาพ annotation 99 ภาพและ negative controls 4 ภาพ รวม 103 ภาพ มี 2,027 วัตถุ ครบ 47 คลาส และภาพหลายคลาส 76 ภาพ ไม่มีการเปลี่ยนโมเดล prompt หรือ threshold หลังเห็นผล test กลุ่มชื่อไฟล์ใช้ส่วนก่อน `.rf.`; การตัดออกแบบนี้อาจตัดภาพที่ต่างกันแต่ใช้ชื่อทั่วไปเหมือนกัน และยังไม่รับประกันการแยกฉากหรือ near duplicates อย่างสมบูรณ์

ไม่ใช้ข้อมูล train ของโครงการนี้เพื่อปรับน้ำหนัก จึงไม่มีการวัด training/validation loss gap หรืออ้างว่าป้องกัน overfitting ของการเทรนแล้ว การแยก validation/test มีไว้ลดอคติในการเลือกโมเดล ส่วนการทับซ้อนกับข้อมูลที่ผู้สร้างโมเดลเคยใช้ pretrained ยังไม่ทราบแน่ชัด ควรเก็บภาพภายนอกที่ถ่ายเองเพิ่มเพื่อทดสอบความสามารถทั่วไป

## วิธีประเมิน

ระบบ inference ใช้ confidence threshold 0.25, NMS IoU 0.45 และภาพเข้าโมเดลขนาด 640 ส่วนการวัดผลใช้ IoU 0.50 เพื่อจับคู่ prediction กับ ground truth ที่มีชื่อคลาสตรงกัน เรียง prediction ตาม confidence และจับคู่แบบหนึ่งต่อหนึ่ง prediction ซ้ำหรือชื่อผิดเป็น FP และ ground truth ที่ไม่พบเป็น FN

- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2TP / (2TP + FP + FN)
- Micro รวมจำนวนวัตถุทั้งหมดก่อนคำนวณ; Macro เฉลี่ยแต่ละคลาสอย่างเท่าเทียม โดยตั้งค่า denominator ที่เป็นศูนย์เป็น 0 ในการเฉลี่ย

ค่าความเชื่อมั่นรายกรอบไม่ได้เป็นความแม่นยำที่ผ่าน calibration และ precision/recall/F1 ในรายงานนี้ไม่ใช่ mAP เพราะ API คืน prediction ที่ผ่าน threshold แล้ว ไม่ได้วัด AP ตลอดช่วง confidence

## เปรียบเทียบ 4 โมเดลบน validation ชุดเดียวกัน

| โมเดล pretrained | TP | FP | FN | Micro Precision | Micro Recall | Micro F1 |
|---|---:|---:|---:|---:|---:|---:|
| YOLO-World v2 small, vocabulary 47 ชื่อ | 51 | 190 | 1,848 | 21.16% | 2.69% | 4.77% |
| YOLO-World v2 medium, vocabulary 47 ชื่อ | 76 | 112 | 1,823 | 40.43% | 4.00% | 7.28% |
| YOLOE 11 small, vocabulary 47 ชื่อ | 67 | 127 | 1,832 | 34.54% | 3.53% | 6.40% |
| YOLOv8m Fruits & Vegetables ของผู้สร้าง | 619 | 627 | 1,280 | 49.68% | 32.60% | 39.36% |

เลือก YOLOv8m Fruits & Vegetables เพราะมี Micro F1 รวมทั้ง 47 คลาสบน validation สูงที่สุด สูงกว่า YOLO-World medium 32.08 จุดเปอร์เซ็นต์ โมเดลต้นฉบับมี 63 labels แต่ map synonym ตรงกับ taxonomy ของ dataset ได้ 42 คลาส กรอง label ตัวพิมพ์ใหญ่ Tomato/Strawberry ที่ซ้ำตามหมายเหตุผู้สร้าง ส่วน 5 คลาสที่ขาดคือ beans, beet, cabbage, egg และ pattypan squash จึงต้องแสดงข้อจำกัดนี้บนเว็บและรายงาน

การกำหนด vocabulary 47 ชื่อให้ YOLO-World/YOLOE ไม่ได้พิสูจน์ว่าตรวจได้ครบทุกชนิดจริง ส่วนโมเดลที่เลือกพบ true positive ใน validation 35 คลาสเท่านั้น

## ผล test หลังตรึงโมเดล

ทุกภาพทดสอบตอบกลับสำเร็จ 103/103 ภาพ ไม่พบ API error มี TP 729, FP 613 และ FN 1,298

| ขอบเขตคะแนน | วิธีเฉลี่ย | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| ทุกคลาสใน dataset 47 คลาส | Micro | 54.32% | 35.96% | 43.28% |
| ทุกคลาสใน dataset 47 คลาส | Macro | 48.05% | 32.88% | 35.27% |
| เฉพาะ 42 คลาสที่โมเดลมี label map | Micro | 54.32% | 37.56% | 44.41% |
| เฉพาะ 42 คลาสที่โมเดลมี label map | Macro | 53.77% | 36.80% | 39.46% |

คะแนนเฉพาะ supported classes เป็นข้อมูลประกอบ และไม่ควรใช้แทนคะแนนทั้ง dataset พบ true positive จริง 39 คลาสจาก 47 คลาส หรือ 39 คลาสจาก 42 คลาสที่มี label map; 3 คลาสที่โมเดลรองรับแต่ไม่พบ TP ใน test ชุดนี้คือ eggplant, hot pepper และ vegetable marrow ไม่รับรองว่าทุกภาพของ 39 คลาสจะถูกต้อง

คะแนน test 43.28% F1 กับ validation 39.36% F1 เกิดจากข้อมูลคนละชุด ไม่ได้แสดงว่าโมเดลเรียนรู้หรือดีขึ้นระหว่างสองรอบ ค่า latency ที่ server วัดเฉพาะ inference เฉลี่ย 73.59 ms, median 70.1 ms, p95 82.1 ms ไม่รวมเวลา upload, decode, annotation และการแสดงผลหน้าเว็บ

Negative controls ทั้ง 4 ภาพไม่มี detection ที่เกิน threshold แต่เป็นภาพสีเรียบ/เรขาคณิต ไม่ใช่หลักฐานว่าไม่มี false positive ในภาพสิ่งของทั่วไป ต้องเก็บชุดภาพธรรมชาติที่ไม่มีผักผลไม้เพิ่มเติม

## ตัวอย่างผลสำเร็จและความผิดพลาด

ตัวอย่างต่อไปนี้เป็นภาพที่เลือกอธิบายผลหลังการประเมิน ไม่ใช่ชุดข้อมูลใหม่หรือคะแนนเพิ่มเติม ทุกภาพอยู่ใน `data/raw/test/images` และผลรายกรอบอยู่ใน `artifacts/pretrained_evaluation.json`

| ไฟล์ภาพ | สิ่งที่พบ | TP / FP / FN |
|---|---|---|
| `-1-22_jpg.rf.9059a7b5d273101fd683cbbd3c708f49.jpg` | broccoli, confidence 0.9161 | 1 / 0 / 0 |
| `-39_jpg.rf.7672d5e2ab7109a11082ede876cb91c5.jpg` | brussels sprouts 6 วัตถุ | 6 / 0 / 0 |
| `-44_jpg.rf.4c5dbd07276f4bbbda10f8b21da6a970.jpg` | cauliflower, confidence 0.8018 | 1 / 0 / 0 |
| `000000086011_jpg.rf.368cce8e80697cfa0d2ba71230bee423.jpg` | apple 2 วัตถุ | 2 / 0 / 0 |
| `000000090662_jpg.rf.218867047ccdc934b893e0d80acac08d.jpg` | banana, confidence 0.9570 | 1 / 0 / 0 |
| `-43_jpg.rf.15825a93b12f850596e89e7a00b9a60b.jpg` | ภาพพริกและกระเทียม; ทำนาย strawberry 3 กรอบและ hot pepper 1 กรอบ แต่ไม่มีคู่ชื่อ/IoU ที่ผ่านเกณฑ์ | 0 / 4 / 3 |
| `000000289497_jpg.rf.fd65dbaf68fe9b14e5017d71b29d1dc9.jpg` | ฉากผลไม้หนาแน่น 236 annotation; พลาดวัตถุเล็กจำนวนมาก | 11 / 10 / 225 |

## การตรวจสอบการทำงานของเว็บ/API

ผ่าน HTTP contract tests 4/4 โดยใช้ synthetic fixtures เท่านั้น ไม่รันทดสอบภาพ final test ซ้ำ:

| กรณี | ผลที่คาดและผลจริง |
|---|---|
| ส่งข้อมูลที่ไม่ใช่ภาพ | HTTP 400 พร้อม error JSON |
| ส่ง Origin ภายนอก localhost | HTTP 403 ก่อน inference |
| Content-Length เกิน 12 MB โดยยังไม่ส่ง body | HTTP 413 ตอบกลับได้ทันที |
| ส่ง negative PNG สำหรับ evaluation | HTTP 200, detection ว่าง, JSON schema ถูกต้อง, รูป JPEG base64 เปิดได้ และ history ไม่เปลี่ยน |

Unit tests สำหรับการจับคู่กรอบและสูตรคะแนนผ่าน 3/3 ครอบคลุม prediction ซ้ำ, คลาสผิด, IoU ไม่ถึงเกณฑ์ และ denominator เป็นศูนย์ การผ่าน software tests ยืนยันการทำงานของระบบ แต่ไม่ได้ยืนยันความแม่นยำของ AI ว่าสูงพอทุกสถานการณ์

## ข้อจำกัดและการพัฒนาต่อ

ผลนี้เป็นตัวอย่าง stratified ขนาดเล็ก ไม่ใช่คะแนนทั้ง 4,619 validation/3,358 test ภาพ ภาพตลาดที่มีวัตถุขนาดเล็กจำนวนมาก, ภาพ collage/อาหารปรุงแล้ว, ภาพเบลอ และชื่อคลาสใกล้เคียงยังทำให้เกิดการพลาดและทำนายผิด จึงควรตรวจผลด้วยสายตาและไม่ใช้ confidence เป็นหลักฐานความถูกต้องเพียงอย่างเดียว

แนวทางต่อยอดคือเก็บภาพจากสถานการณ์ใช้งานจริงพร้อมภาพ negative ธรรมชาติ, ตรวจคุณภาพ label และข้อมูลซ้ำ, ประเมิน full split และ AP ด้วย prediction ที่ไม่ถูกตัด confidence, ทดสอบ calibration ด้วย validation แยก, และหากต้องการ fine tune ในอนาคตให้จัด experiment ใหม่พร้อมแยก train/validation/external test, early stopping และบันทึก learning curves โดยขอเปลี่ยนขอบเขตจากโหมดไม่เทรนก่อน

## หลักฐานที่บันทึกและการทำซ้ำ

- `artifacts/pretrained_validation_manifest.json`: รายการภาพ validation และ hashes
- `artifacts/pretrained_evaluation_manifest.json`: รายการภาพ test และการตัด source groups
- `artifacts/world_small_validation.json`, `world_medium_validation.json`, `yoloe_small_validation.json`, `domain_validation.json`: รายภาพและคะแนนทั้ง 4 โมเดล
- `artifacts/pretrained_evaluation.json`: ผล test สุดท้าย พร้อม prediction, truth, confidence, คู่ IoU และ latency
- `artifacts/pretrained_comparison.json`: ตารางเปรียบเทียบ, Micro/Macro, supported-class subset และกฎเลือกโมเดล
- `artifacts/api_checks.json`: ผลตรวจ HTTP contract
- `tests/evaluate_pretrained.py`, `tests/summarize_pretrained.py`, `tests/test_pretrained_evaluation.py`, `tests/test_http_api.py`: สคริปต์ทำซ้ำ; อ่านคำสั่งใน `tests/README.md`

แหล่งโมเดล: [ผู้สร้าง Fruits And Vegetables Detection Dataset](https://github.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset), [Ultralytics YOLO-World](https://docs.ultralytics.com/models/yolo-world/), [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/) รายงานนี้อ้างคะแนนที่วัดจากเครื่องนี้เท่านั้น ไม่แทนคะแนน benchmark ของผู้สร้าง
