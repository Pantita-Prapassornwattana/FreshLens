# แหล่งข้อมูลและโมเดลสำหรับการตรวจจับโดยไม่เทรน

บันทึกตรวจสอบแหล่งข้อมูลวันที่ 7 ตุลาคม 2026 (เวลาไทย) ตามคำขอให้ใช้โมเดลสำเร็จรูปและไม่เทรนเพิ่ม เอกสารนี้แสดงแหล่งอ้างอิงและเหตุผลในการทดลองแต่ละโมเดล โมเดลที่เว็บเลือกใช้งานจริงให้ตรวจจาก `artifacts/selected_model.json` และผลประเมินใน `artifacts/` ซึ่งอาจเปลี่ยนหลังเปรียบเทียบตัวเลือกเพิ่มเติม

## Dataset ของโครงการ

ใช้ [Combined Vegetables & Fruits v8 ของ Yolo บน Roboflow](https://universe.roboflow.com/yolo-jpkho/combined-vegetables-fruits/dataset/8) ซึ่งผู้ใช้ดาวน์โหลดมาแล้ว เก็บต้นฉบับ ZIP และข้อมูลแตกไฟล์ไว้ภายในโครงการ

| รายการ | ข้อมูลจากผู้เผยแพร่ |
|---|---:|
| จำนวนภาพทั้งหมด | 41,985 |
| Train | 34,008 |
| Validation | 4,619 |
| Test | 3,358 |
| รูปแบบ annotation ที่ดาวน์โหลด | YOLO TXT และ YAML |
| การเตรียมภาพ | Auto-orient และ stretch เป็น 416 × 416 |
| Augmentation ในรุ่นนี้ | ไม่มี |
| ใบอนุญาตที่หน้า Dataset ระบุ | CC BY 4.0 |

ไฟล์ `data/raw/data.yaml` ระบุ 47 คลาส รวม `egg` และ `almond` ดังนั้นชื่อ Dataset ไม่ได้หมายความว่าทุกคลาสเป็นผักหรือผลไม้ทางพฤกษศาสตร์ จำนวนใช้งานจริง ไฟล์ผิดรูปแบบ และรายการ exclusion ให้ยึด inventory และ evaluation manifests ที่ตรวจในเครื่อง

ในโหมดนี้ Dataset ใช้เป็นภาพตัวอย่างและข้อมูลประเมิน ไม่ใช้ปรับน้ำหนักโมเดล ตั้งแต่ผู้ใช้ขอให้ไม่เทรนจึงต้องบันทึก epochs ใหม่เป็น 0 และไม่อ้างว่ามีการ fine-tune บนข้อมูลชุดนี้

## ตัวเลือก A: YOLO-World v2 กับ CLIP

อ้างอิง [เอกสาร Ultralytics YOLO-World](https://docs.ultralytics.com/models/yolo-world), [ต้นฉบับ YOLO-World ของ AILab-CVC](https://github.com/AILab-CVC/YOLO-World) และ [งานวิจัย CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.pdf)

ใช้ checkpoint สำเร็จรูป `yolov8s-worldv2.pt` หรือ `yolov8m-worldv2.pt` จาก Ultralytics assets ผ่าน `YOLOWorld(...)` จากนั้นเรียก `set_classes()` ด้วยชื่อภาษาอังกฤษตาม Dataset และบันทึก checkpoint ที่มี vocabulary embeddings ภายในโครงการ ขั้นตอนนี้คำนวณ embeddings สำหรับ inference ไม่ใช่การเทรนหรือ prompt tuning

```python
from ultralytics import YOLOWorld

model = YOLOWorld("models/yolov8s-worldv2.pt")
model.set_classes(class_names)
model.save("models/yolov8s-worldv2-produce.pt")
results = model.predict("example.jpg", conf=0.25, imgsz=640)
```

ระบบมี CNN backbone สำหรับภาพ, detection head สำหรับตำแหน่ง และการจับคู่ region features กับข้อความ CLIP เพื่อเลือกประเภท อ้างอิง CLIP จาก [repository ของ OpenAI](https://github.com/openai/CLIP) และ implementation ที่ใช้จาก [Ultralytics CLIP](https://github.com/ultralytics/CLIP)

การประกาศ 47 prompts แสดงว่าระบบสามารถรับรายชื่อ 47 ประเภท ไม่รับรองว่าจะตรวจถูกทุกประเภท โดยเฉพาะชื่อใกล้กัน เช่น beans/green bean, mandarin/orange และผักที่มีรูปร่างคล้ายกันต้องประเมินรายคลาส

### หลักฐาน local validation ที่ตรวจได้แล้ว

ตารางนี้เป็น snapshot ของผลที่อ่านจาก `artifacts/world_small_validation.json` และ `artifacts/world_medium_validation.json` เป็นการตรวจจับที่ confidence 0.25 และ IoU 0.50 บน validation 108 ภาพชุดเดียวกัน จับคู่ชนิดและกรอบตรงกับ annotation จึงนับเป็น TP คะแนนนี้ไม่ใช่ accuracy ของภาพและไม่ใช่ COCO mAP

| Candidate | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| World v2 small | 51 | 190 | 1,848 | 21.16% | 2.69% | 4.77% |
| World v2 medium | 76 | 112 | 1,823 | 40.43% | 4.00% | 7.28% |

ผล medium ดีกว่า small ในชุดย่อยนี้ แต่ recall ยังต่ำมาก จึงยังไม่เป็นหลักฐานว่าตรวจครบ 47 ประเภทอย่างน่าเชื่อถือ ต้องเปรียบเทียบตัวเลือกอื่นก่อนสรุป ไม่มีการเทรนเพื่อให้เกิดการปรับปรุงในตารางนี้ ความต่างเกิดจาก checkpoint สำเร็จรูปคนละขนาด

## ตัวเลือก B: YOLOE 11 small กับ MobileCLIP

อ้างอิง [เอกสาร Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe) และ [โครงการต้นฉบับ THU-MIG](https://github.com/THU-MIG/yoloe)

ใช้ `YOLOE("yoloe-11s-seg.pt")` และ `set_classes(class_names)` เป็นอีกตัวเลือกสำหรับ zero-shot detection/segmentation การตั้งข้อความครั้งแรกต้องมี CLIP tokenizer และ MobileCLIP encoder `mobileclip_blt.ts` จากนั้นเก็บ embeddings กับ checkpoint ไว้เพื่อ inference แบบ local ตรวจจับได้โดยไม่เทรนเพิ่ม ใช้ `*-seg.pt` เมื่อกำหนดข้อความเอง เพราะ `*-seg-pf.pt` ไม่รองรับ `set_classes()`

ผล validation/test ของ YOLOE ให้ยึดไฟล์ประเมินที่มี `complete: true` และชื่อ checkpoint ตรงกับ candidate ไม่มีการนำตัวเลขจาก benchmark ของผู้เผยแพร่มาแทนผลบน Dataset ของผู้ใช้

## ตัวเลือก C: โมเดลผักผลไม้ 63 คลาสที่เทรนมาแล้ว

พบ [โครงการต้นฉบับ Fruits-And-Vegetables-Detection-Dataset ของ Henning Heyen](https://github.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset) ที่เผยแพร่ baseline YOLOv8 หลายขนาด ผู้สร้างรายงาน Dataset จาก LVIS 8,221 ภาพ และชุด test เพิ่ม 180 ภาพที่ annotate เอง ภาพและประเภทในชุดนี้มีข้อจำกัดเรื่อง class imbalance และ upper/lower case ของ Tomato/Strawberry ผู้สร้างระบุว่าหมวดตัวอักษรใหญ่มีภาพที่ไม่ตรงประเภทบางส่วน

น้ำหนักต้นฉบับมีใน [Google Drive ที่ผู้สร้างลิงก์ไว้ใน README](https://drive.google.com/drive/folders/1I4mtQK11C3p41pO9raR0trgPVj0eQ2yb?usp=sharing) ตรวจรายชื่อไฟล์จาก public folder ได้ดังนี้:

| ชื่อไฟล์ | Drive file ID | ขนาดที่หน้า Drive ระบุ |
|---|---|---:|
| yolo_fruits_and_vegetables_v1.pt | 1XALsPwM4850LkmqsXIfmGhPj8s_nhuiK | 49.7 MB |
| yolo_fruits_and_vegetables_v2.pt | 1EntTqiFlY9pQmxUuFKdzhorkt8sYUko6 | 83.7 MB |
| yolo_fruits_and_vegetables_v3.pt | 1aaRDJWTYZKBEnt9wq719C9AKi8JskWDP | 130.5 MB |

ตัวเลข v1/v2/v3 และขนาดไฟล์ไม่ใช่หลักฐานของสถาปัตยกรรมที่แน่นอน ต้องอ่าน checkpoint metadata หลังดาวน์โหลดก่อนระบุว่ารุ่นใดเป็น medium/large/xlarge

พบ mirror [Senu-12/snapstock-fruit-vegetable-detector](https://huggingface.co/Senu-12/snapstock-fruit-vegetable-detector) ซึ่งระบุชัดว่ารับน้ำหนัก YOLOv8m จาก Henning Heyen และไม่ได้อ้างเป็นผู้สร้างต้นฉบับ ไฟล์อยู่ที่ `yolov8/fruit_vegetable_yolov8m.pt` การดาวน์โหลด mirror ผ่านเครื่องในช่วงตรวจสอบพบข้อผิดพลาด SSL จึงไม่ถือว่าดาวน์โหลดหรือใช้งานสำเร็จจนมี checkpoint จริง

ชนิดของโมเดล 63 คลาสไม่ตรงกับ 47 คลาสของ Roboflow โดยตรง บางประเภทไม่มี เช่น beet, cabbage, egg, beans และ pattypan squash ต้องใช้ mapping ที่ประกาศชัด และห้ามเปลี่ยนความหมายของคลาสเพื่อให้คะแนนดูดี ส่วน synonym เช่น `cucumber/cuke` รวมถึง `mandarin orange` สามารถ normalize โดยเก็บชื่อดั้งเดิมประกอบผลไว้

คะแนน mAP50-95 0.152 ของ medium ถึง 0.202 ของ xlarge เป็นตัวเลขที่ผู้สร้างรายงานบนข้อมูลของเขา ไม่ใช่ผลทดสอบบน Roboflow v8 ของโครงการนี้ การยืนยันว่าโมเดลนี้ดีกว่า World/YOLOE ต้องอาศัย local evaluation ชุดเดียวกัน

## เทคนิคที่อธิบายได้ตามการทำงานจริง

| เทคนิค | ใช้ในส่วนใด |
|---|---|
| Classification | เลือกชนิดผักผลไม้ให้แต่ละกรอบตรวจจับ |
| Regression | พยากรณ์พิกัดกรอบ bounding box ด้วย detection head สำเร็จรูป |
| Neural network และ Deep learning | ประมวลผลภาพและ feature/embedding ด้วย pretrained neural models |
| CNN | backbone และ layers ของ YOLO ดึงลักษณะจากภาพ |
| NLP / text embeddings | CLIP หรือ MobileCLIP แปลงชื่อภาษาอังกฤษเป็น vector เพื่อจับคู่กับภาพ ไม่ใช่แชตบอต |
| Evaluation / Optimization | เปรียบเทียบ checkpoint และ threshold ด้วย validation แล้วตรึงค่าเพื่อประเมิน test |

Bayesian inference, unsupervised learning, dimensionality reduction, RNN และ RL ยังไม่ใช่ส่วนที่พิสูจน์ว่าใช้ใน inference นี้ หากมีโค้ดหรือแผนทดลองเดิมอยู่ ต้องระบุว่าเป็นแผน/การทดลองเสริมที่ไม่ได้รันในโหมดไม่เทรน การแสดง confidence เพียงอย่างเดียวไม่ใช่หลักฐานว่าได้ใช้ Bayesian inference

การไม่เทรนเพิ่มไม่ได้ทำให้ปัญหา overfitting หรือ data leakage หายไป: checkpoint อาจเคยเห็นภาพสาธารณะคล้ายกันก่อนหน้า และการเปลี่ยน prompts/threshold โดยดู test ซ้ำทำให้ประเมินเอนเอียงได้ จึงใช้ validation เลือกค่าและเก็บ test สำหรับผลหลัง freeze พร้อมบันทึก manifest, image hash, label hash, checkpoint และ threshold ภาพ synthetic negative controls ทดสอบภาพว่าง/รูปทรงเท่านั้น ไม่แทนภาพจริงที่ไม่มีผักผลไม้

## ใบอนุญาตและการอ้างอิง

| องค์ประกอบ | แหล่งใบอนุญาตที่ตรวจได้ |
|---|---|
| Roboflow Combined Vegetables & Fruits v8 | CC BY 4.0 ตามหน้า Dataset; อ้าง Yolo และ Roboflow พร้อม URL/version |
| Ultralytics runtime | [AGPL-3.0 ของ repository](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) |
| YOLO-World ต้นฉบับ | [GPL-3.0](https://github.com/AILab-CVC/YOLO-World/blob/master/LICENSE) |
| OpenAI CLIP ต้นฉบับ | [MIT](https://github.com/openai/CLIP/blob/main/LICENSE) |
| Henning Heyen project code | [MIT ปี 2024](https://github.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset/blob/main/LICENSE) |
| HF mirror ของโมเดล 63 คลาส | Model card ระบุ license: other; เก็บ attribution ต้นฉบับและไม่อ้างว่า mirror ให้สิทธิ์ครอบคลุมภาพ LVIS ทั้งหมด |

ตารางใบอนุญาตเป็นบันทึกแหล่งที่ใช้ตรวจสอบ ไม่ควรสรุปว่า MIT ของโค้ดโครงการครอบคลุมสิทธิ์ภาพทุกภาพหรือ checkpoint ทุกแหล่งโดยอัตโนมัติ หากนำ Dataset หรือโมเดลไปเผยแพร่ต่อ ให้เก็บ notice และอ้างอิงที่มากับองค์ประกอบนั้น

