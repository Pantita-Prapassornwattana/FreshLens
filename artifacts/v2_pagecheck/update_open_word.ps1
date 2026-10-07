$ErrorActionPreference = 'Stop'
$target = 'C:\Users\Acer\Desktop\Mini_Project_FreshLens_Ready_to_SubmitV2.docx'
$fix = [ordered]@{
  'บทคัดย่อ'=4; 'ขอบเขตของรายงาน'=4; '1. ที่มาและความสำคัญ'=5; 'วัตถุประสงค์ของโครงการ'=5; 'ขอบเขต'=5;
  '2. Dataset และการเตรียมข้อมูล'=6; 'การเตรียมภาพก่อน inference'=6;
  '3. โมเดล YOLO ที่ใช้และเหตุผลในการเลือก'=7; 'โมเดลหลักของระบบ'=7;
  '4. โมเดลเสริมและการทำงานของระบบตรวจจับ'=9; 'กติกาการรวมผล'=9;
  '5. เทคนิค AI ที่ใช้จริงในโครงการ'=11; 'Neural Network และ Deep Learning'=11; 'CNN: Convolutional Neural Network'=11;
  'Classification: การเลือกชนิดของวัตถุ'=11; 'Regression: การหาพิกัดกรอบ'=11; 'การประเมินและการปรับ configuration'=11;
  '6. สถาปัตยกรรมเว็บและลำดับการใช้งาน'=12; 'ขั้นตอนผู้ใช้'=12; 'การจัดการข้อผิดพลาดและข้อจำกัดอินพุต'=12; 'เครื่องมือและสภาพแวดล้อม'=13;
  '7. วิธีประเมินโมเดลและความหมายของตัวชี้วัด'=13; '8. ผลการทดลองและการวิเคราะห์ผล'=14; 'ทำไมต้องดูหลายค่า'=14;
  'ผลนี้บอกอะไร'=15; 'โมเดลฉลาดขึ้นอย่างไร'=15; '9. การทดลองที่ไม่เลือกและผลของการเปรียบเทียบ'=16;
  '10. ตรวจสอบการทำงานของเว็บ'=17; 'เหตุใดคะแนน validation กับ final test ต่างกัน'=16; 'เวลาประมวลผล'=17;
  '11. ความน่าเชื่อถือและข้อจำกัดของผล'=18; 'งานพัฒนาต่อที่แนะนำ'=19;
  '12. สรุปและข้อเสนอแนะพัฒนาต่อ'=19; 'เอกสารอ้างอิงและตำแหน่งหลักฐาน'=20; 'อภิธานศัพท์ในรายงาน'=20
}
$word = [Runtime.InteropServices.Marshal]::GetActiveObject('Word.Application')
$doc = $null
foreach ($d in $word.Documents) { if ($d.FullName -ieq $target) { $doc = $d; break } }
if (-not $doc) { throw 'Target V2 document is not open in the active Word instance.' }
$changed = 0
for ($i = $doc.Paragraphs.Count; $i -ge 1; $i--) {
  $para = $doc.Paragraphs.Item($i)
  $txt = $para.Range.Text.TrimEnd([char]13, [char]7)
  $tab = $txt.LastIndexOf("`t")
  if ($tab -lt 0) { continue }
  $title = $txt.Substring(0, $tab)
  if (-not $fix.Contains($title)) { continue }
  $start = $para.Range.Start + $tab + 1
  $end = $para.Range.End - 1
  if ($end -lt $start) { throw "Bad range in TOC entry: $title" }
  $doc.Range($start, $end).Text = [string]$fix[$title]
  $changed++
}
if ($changed -ne $fix.Count) { throw "Updated $changed of $($fix.Count) TOC entries." }
$doc.Save()
Write-Output "Updated and saved $changed TOC entries in $($doc.Name)."
