# แหล่งโมเดลเพิ่มเติมสำหรับผักที่ยังตรวจไม่พบ

ตรวจสอบวันที่ 7 ตุลาคม 2026 ตามคำขอให้เพิ่มความสามารถในการตรวจผักโดยใช้โมเดลที่ผู้อื่นเทรนมาแล้ว ไม่มีการเทรนหรือ fine-tune ใหม่ในขั้นตอนค้นหาแหล่งเหล่านี้ เอกสารนี้เป็นหลักฐานการค้นหา ไม่ใช่ผลรับรองความแม่นยำของโมเดลบนเครื่องผู้ใช้

## ตัวเลือกที่เพิ่มประเภทผักได้ตรงปัญหา

พบ YOLO11 สอง checkpoint ที่ผู้เผยแพร่ระบุ 35 คลาส จาก Dataset `bohni-tech/fruits-and-vegi` ทั้งคู่มีประเภทผักซึ่งโมเดล LVIS 63 คลาสเดิมขาด เช่น cabbage และ beetroot รวมทั้งผักที่พบในตลาดไทยบางชนิด:

| ชื่อที่ model card ระบุ | ชื่อแสดงผลภาษาไทยที่เหมาะสม | การใช้ร่วมกับ Roboflow v8 เดิม |
|---|---|---|
| Beetroot | บีตรูต | map เป็น beet |
| Cabbage | กะหล่ำปลี | map เป็น cabbage |
| EggPlant | มะเขือยาว | map เป็น eggplant; ไม่รับรองมะเขือทุกพันธุ์ |
| Chilli | พริก | map เป็น hot pepper; ไม่รับรองพันธุ์/ระดับความเผ็ด |
| Ginger | ขิง | เป็นคลาสเพิ่มเติมนอก 47 คลาสเดิม |
| Okra | กระเจี๊ยบเขียว | เป็นคลาสเพิ่มเติมนอก 47 คลาสเดิม |
| Bitter_Gourd | มะระ | เป็นคลาสเพิ่มเติมนอก 47 คลาสเดิม |
| Bottle_Gourd | น้ำเต้า | เป็นคลาสเพิ่มเติมนอก 47 คลาสเดิม |

ประเภทเพิ่มเติมต้องมีภาพประเมินแยก เนื่องจาก test ของ Roboflow v8 เดิมไม่มี annotation สำหรับขิง กระเจี๊ยบเขียว มะระ และน้ำเต้า ห้ามสรุปคะแนนของประเภทใหม่โดยใช้คะแนนรวมของ 47 คลาสเดิมแทน

### Piyu12: YOLO11 medium

แหล่งต้นทางคือ [model card ของ Piyu12](https://huggingface.co/Piyu12/fruit-veg-yolo11m-detector) และ [Files and versions](https://huggingface.co/Piyu12/fruit-veg-yolo11m-detector/tree/main) เป็นโมเดล detector 35 คลาส ผู้เผยแพร่ระบุว่าปรับจาก YOLO11m และใช้ Bohni Tech รุ่น v13 ภาพขนาด 640 × 640 ใบอนุญาตใน card เป็น AGPL-3.0

ไฟล์ที่ยืนยันจากหน้า Files คือ `best.pt` ที่ root ของ repository ขนาด 40.6 MB:

- [ดาวน์โหลด best.pt ของ Piyu12](https://huggingface.co/Piyu12/fruit-veg-yolo11m-detector/resolve/main/best.pt)
- API: `hf_hub_download("Piyu12/fruit-veg-yolo11m-detector", "best.pt", local_dir="models/...")` แล้วโหลดผ่าน `ultralytics.YOLO(path)`

ผู้เผยแพร่รายงาน mAP50 0.925 และ mAP50-95 0.691 บนข้อมูลของผู้เผยแพร่ ตัวเลขนี้ไม่ใช่ผลทดสอบของโครงการ ไม่ใช้ยืนยันว่าตรวจผักไทยทุกพันธุ์ได้ เหมาะเป็นตัวเลือกแรกก่อนรุ่น large เนื่องจากมีขนาดไฟล์เล็กกว่า แต่ความเหมาะสมกับ GPU และ latency ต้องวัดจริง

ภายหลังนำ checkpoint Piyu12 ลงโครงการสำเร็จผ่าน HTTPS โดยยังตรวจ certificate ตามปกติ และตรวจ metadata ว่ามี 35 คลาสตรงกับ model card:

- repository revision: `82c9865261175e778ba5641cf8560b5d10439ec6`
- ขนาดไฟล์จริง: 40,565,868 bytes
- SHA-256: `499ccee2169731363de26146d9989942c27f3849f64509e0b5190cbd1c679c19`
- [URL ที่ตรึง revision ของ best.pt](https://huggingface.co/Piyu12/fruit-veg-yolo11m-detector/resolve/82c9865261175e778ba5641cf8560b5d10439ec6/best.pt)

checkpoint มี 10 labels ซึ่งไม่มีชื่อคลาสตรงใน Roboflow v8 เดิม: `Bitter_Gourd`, `Bottle_Gourd`, `Coconut`, `Ginger`, `Green_Orange`, `Mango`, `Melon`, `Okra`, `Pomegranate` และ `Turnip` การเพิ่ม 10 labels ไม่ได้แปลว่าเพิ่ม 10 ชนิดพืช เพราะ `Green_Orange` เป็นคำบรรยายส้มสีเขียวที่อาจรวมกับ orange เมื่อ canonical mapping ต้องกำหนดความหมายใน UI ให้ตรงกับ mapping ที่ใช้จริง ส่วน melon เป็นชื่อกว้าง ไม่ควรแปลว่าแคนตาลูปทุกกรณี

ชื่อคลาสใน metadata เป็นหลักฐานว่ามี output สำหรับประเภทดังกล่าว แต่ยังไม่พิสูจน์ความแม่นยำท่ามกลางพันธุ์ผักไทยหรือฉากใช้งานจริง ชนิดที่ไม่มี annotation ใน Dataset เดิมให้แสดงว่า accuracy ยังไม่ยืนยัน จนมีภาพที่ annotate และผลประเมินเฉพาะประเภท

### Aniket2003333333: YOLO11 large

แหล่งต้นทางคือ [model card ของ Aniket2003333333](https://huggingface.co/Aniket2003333333/fruit-veg-yolo11l-detector) และ [Files and versions](https://huggingface.co/Aniket2003333333/fruit-veg-yolo11l-detector/tree/main) ผู้เผยแพร่ระบุ YOLO11l, 35 คลาส และ Bohni Tech Fruits and Vegi เช่นเดียวกัน ใบอนุญาตใน card ระบุ Apache-2.0 ซึ่งเป็น metadata ที่ผู้เผยแพร่ใส่ ไม่ใช่ข้อยืนยันว่าการใช้ Ultralytics runtime เปลี่ยนใบอนุญาตตามไปด้วย

ไฟล์คือ `best.pt` ที่ root ขนาด 51.2 MB:

- [ดาวน์โหลด best.pt ของ Aniket2003333333](https://huggingface.co/Aniket2003333333/fruit-veg-yolo11l-detector/resolve/main/best.pt)
- API: `hf_hub_download("Aniket2003333333/fruit-veg-yolo11l-detector", "best.pt", local_dir="models/...")`

ผู้เผยแพร่รายงาน mAP50 0.913 และ mAP50-95 0.689 ไม่ควรเลือกเพียงเพราะชื่อรุ่นใหญ่กว่า และไม่ควรจัดอันดับจากตัวเลขคนละแหล่งโดยไม่ทดสอบ Dataset/configuration เดียวกัน

ทั้งสอง repository มี README, .gitattributes และ checkpoint แต่ไม่พบลิงก์ Google Drive หรือ GitHub ที่ผู้เผยแพร่ระบุเป็น mirror ของน้ำหนักชุดเดียวกัน ในช่วงค้นหาแรกการอ่านผ่าน web search ได้ แต่การร้องขอผ่าน Python ของเครื่องพบ TLS connection reset (10054) ทั้ง API และ resolve URL การใช้ `hf.co` ให้ redirect กลับ Hugging Face โดยภายหลังดาวน์โหลด Piyu12 รุ่นที่ตรึง revision สำเร็จตามบันทึกด้านบน ไม่มีการปิดตรวจ certificate เพื่อหลีกเลี่ยงข้อผิดพลาด

## ทางเลือกจาก GitHub ที่ดาวน์โหลดได้โดยไม่พึ่ง HF

พบ [Pic2Recipe ของ Ryotess](https://github.com/Ryotess/Pic2Recipe--A_yolov8_based_recipe_recommender) ซึ่งผู้สร้างระบุการใช้ YOLOv8 และ Dataset [Bohni Tech Fruits and Vegi v13](https://universe.roboflow.com/bohni-tech/fruits-and-vegi/dataset/13) เช่นกัน เป็นคนละ checkpoint กับ YOLO11 สองรุ่นข้างต้น

GitHub API tree ยืนยันว่ามี checkpoint จริงที่ `runs/detect/train/weights/best.pt` ขนาด 6,267,495 bytes พร้อม training args และ results.csv; `recipe_web.py` ใช้ `YOLO()` โหลดไฟล์นั้นจริง ขนาดไฟล์อย่างเดียวไม่เพียงพอสำหรับยืนยัน YOLOv8 variant หรือรายชื่อคลาส ต้องอ่าน checkpoint metadata ก่อนนำเข้าระบบ

- [checkpoint best.pt จาก repository ผู้สร้าง](https://raw.githubusercontent.com/Ryotess/Pic2Recipe--A_yolov8_based_recipe_recommender/main/runs/detect/train/weights/best.pt)
- [training args ของผู้สร้าง](https://raw.githubusercontent.com/Ryotess/Pic2Recipe--A_yolov8_based_recipe_recommender/main/runs/detect/train/args.yaml)
- [training results ของผู้สร้าง](https://raw.githubusercontent.com/Ryotess/Pic2Recipe--A_yolov8_based_recipe_recommender/main/runs/detect/train/results.csv)
- commit HEAD ที่ตรวจ: `a53006d8357c43d747ef8d11cbe7a958c7ccb18b` สามารถแทน `main` ใน raw URL เพื่ออ้างไฟล์รุ่นคงที่

อ่านแหล่ง GitHub ผ่าน HTTPS บนเครื่องสำเร็จ จึงเป็น fallback ที่ทำได้จริงสำหรับนำ checkpoint มาประเมิน รายชื่อคลาสยังต้องยืนยันจาก model.names เพราะไม่มี class mapping ใน source ที่อ่าน

ผู้สร้างบันทึก 20 epochs ของงานเดิมบนภาพ 640 × 640 และไฟล์ผลสุดท้ายที่ epoch 19 ระบุ precision 0.84364, recall 0.83037, mAP50 0.89554, mAP50-95 0.67914 ตัวเลขเหล่านี้เป็นผลการเทรนของผู้สร้าง ไม่ใช่การเทรนใหม่ในโครงการนี้ และไฟล์ `best.pt` อาจเป็นคนละ epoch กับแถวสุดท้ายของ CSV ไม่พบ LICENSE จาก repository API ในช่วงตรวจ จึงบันทึกสถานะว่าไม่ได้ประกาศใน repo แทนการอนุมานสิทธิ์จากความเป็น public

## แผนประเมินก่อนเพิ่มในระบบ

1. เก็บ checkpoint และ SHA-256, source URL, repository revision, model.names และเวลาโหลดไว้ในโครงการ
2. ตรวจว่า 8 ชนิดผักในตารางมีชื่อและความหมายตรงจริง ห้ามเปลี่ยน beetroot ให้เป็นผักอื่นหรือรวม gourd ทุกชนิดเข้าด้วยกัน
3. เปรียบเทียบ baseline เดิม, detector ใหม่เดี่ยว และระบบรวม บน validation manifest เดียวกัน รายงาน per-class TP/FP/FN โดยแยก cabbage, beet, eggplant และ hot pepper
4. สำหรับประเภทที่มีในหลายโมเดล ให้กำจัดกรอบซ้ำหลัง canonical label mapping แต่ต้องตรวจ class conflict และภาพที่เป็นพืชใกล้เคียงกัน การลด threshold ต้องดู false positives ประกอบ
5. จัดภาพประเมินที่ annotate เองสำหรับขิง กระเจี๊ยบเขียว มะระ น้ำเต้า และภาพ non-produce จริง ก่อนรับรองประเภทเพิ่มเติม
6. เลือกค่าและโมเดลจาก validation แล้ว freeze config ก่อนประเมิน final test ใหม่ด้วย manifest ที่ไม่ใช้เลือกค่า เก็บผลเดิมไว้เพื่อให้เห็นพัฒนาการ

การใช้โมเดลหลายตัวและ post-processing ยังเป็น inference ไม่มี epochs เพิ่ม การปรับ threshold, mapping, crop หรือ multi-scale ใช้ชื่อว่า configuration optimization และบันทึกเวลา/คะแนนเทียบเดิมเพื่อรายงานอย่างตรงไปตรงมา
