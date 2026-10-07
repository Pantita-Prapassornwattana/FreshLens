'use strict';
(() => {
 const el=id=>document.getElementById(id);
 const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const numeric=value=>typeof value==='number'&&Number.isFinite(value);
 const pct=value=>numeric(value)?(value*100).toFixed(2)+'%':'—';
 const charts=[['boxChart','box_train','box_val'],['clsChart','cls_train','cls_val'],['dflChart','dfl_train','dfl_val'],['mapChart','map50','map50_95'],['prChart','precision','recall']];
 let loading=false;
 function chart(id,rows,keys,labels){
  const values=rows.flatMap(r=>keys.map(k=>r[k])).filter(numeric);
  if(!values.length){el(id).innerHTML='<p class="hint">ยังไม่มีค่าที่วัดจริงสำหรับกราฟนี้</p>';return}
  const percentage=id==='mapChart'||id==='prChart',max=Math.max(...values,percentage?1:0.01)*1.05;
  const epochs=rows.map(r=>r.epoch),lo=Math.min(...epochs),hi=Math.max(...epochs);
  const x=e=>42+(e-lo)/Math.max(1,hi-lo)*306,y=v=>170-v/max*135;
  const colors=['#205b40','#d59b32'];
  let svg='<svg viewBox="0 0 390 205" role="img" aria-label="กราฟค่าที่วัดจริงราย epoch"><path d="M42 20V170H355" fill="none" stroke="#dce4da"/>';
  for(let i=0;i<=4;i++){const v=max*i/4;svg+=`<text x="36" y="${y(v)+4}" text-anchor="end" font-size="10" fill="#64736a">${percentage?(v*100).toFixed(0)+'%':v.toFixed(2)}</text>`}
  keys.forEach((key,index)=>{
   const points=rows.filter(r=>numeric(r[key]));
   // Missing metrics break the line instead of inventing intermediate observations.
   let path='',previous=-2;
   rows.forEach((r,j)=>{if(numeric(r[key])){path+=`${j===previous+1?'L':'M'}${x(r.epoch)} ${y(r[key])} `;previous=j}else previous=-2});
   svg+=`<path d="${path}" fill="none" stroke="${colors[index]}" stroke-width="2"/>`;
   points.forEach(r=>{svg+=`<circle cx="${x(r.epoch)}" cy="${y(r[key])}" r="3" fill="${colors[index]}"/>`});
  });
  svg+=`<text x="42" y="190" font-size="11">${lo}</text><text x="350" y="190" text-anchor="end" font-size="11">${hi}</text><text x="196" y="201" text-anchor="middle" font-size="10">Epoch</text></svg>`;
  el(id).innerHTML=svg+`<div class="chart-legend">${labels.map((l,i)=>`<span><i style="background:${colors[i]}"></i>${escape(l)}</span>`).join('')}</div>`;
 }
 function comparison(data){
  if(!data?.baseline||!data?.upgraded){el('trainingComparison').textContent='ยังไม่มีผลเปรียบเทียบที่วัดจริง';return}
  let html=`<p>โมเดลหลักเทียบกับระบบรวมโมเดล บน ${escape(data.images)} ภาพเดียวกัน<br>เป็นผลปรับระบบตรวจจับ ไม่ใช่ผลเทรนราย epoch</p>`;
  for(const [key,name] of [['precision','Precision'],['recall','Recall'],['f1','F1']]){
   html+=`<strong>${name}</strong>`;
   for(const [label,values,color] of [['โมเดลหลัก',data.baseline,'#a7bd97'],['ระบบเสริม',data.upgraded,'#205b40']]){
    if(numeric(values[key]))html+=`<div class="comparison-bar"><span>${label}</span><div><i style="width:${Math.max(0,Math.min(100,values[key]*100))}%;background:${color}"></i></div><b>${pct(values[key])}</b></div>`;
   }
  }
  el('trainingComparison').innerHTML=html;
  el('trainingComparisonDetails').textContent=JSON.stringify(data,null,2);
 }
 async function refreshTraining(){
  if(loading)return;loading=true;
  try{
   const response=await fetch('/api/training',{signal:AbortSignal.timeout(10000)});
   if(!response.ok)throw Error(`อ่านสถานะไม่สำเร็จ (HTTP ${response.status})`);
   const data=await response.json(),rows=Array.isArray(data.rows)?data.rows:[];
   const names={not_started:'ยังไม่ได้เริ่มเทรนเพิ่ม จึงยังไม่มีกราฟการเรียนรู้ราย epoch',running:'สถานะที่บันทึกล่าสุด: กำลังเทรน',completed:'เทรนเสร็จแล้ว',failed:'การเทรนหยุดเนื่องจากข้อผิดพลาด',recorded:'มีผลราย epoch ที่บันทึกไว้'};
   el('trainingStatus').textContent=names[data.status]||`สถานะที่บันทึก: ${data.status}`;
   el('trainingStatus').classList.toggle('ready',rows.length>0);
   el('trainingError').textContent=data.error||'';
   el('trainingEpoch').textContent=`${data.completed_epochs??0} / ${data.requested_epochs??'—'}`;
   el('trainingMap').textContent=pct(data.latest?.map50_95);el('trainingRecall').textContent=pct(data.latest?.recall);
   el('trainingProgress').max=Math.max(1,data.requested_epochs??data.completed_epochs??1);el('trainingProgress').value=data.completed_epochs??0;
   el('trainingCounts').textContent=data.run_id?`Run: ${data.run_id} · ภาพฝึก ${data.train_images??'ไม่ระบุ'} ภาพ · บันทึก ${rows.length} epoch`:'ยังไม่มีชุดข้อมูลที่ใช้ปรับน้ำหนักในรอบนี้ กราฟผลเปรียบเทียบระบบที่วัดจริงแสดงด้านล่าง';
   el('trainingSettings').innerHTML=Object.entries(data.settings||{}).map(([key,v])=>`<span>${escape(key)}: ${escape(v)}</span>`).join('');
   charts.forEach(([id,...keys])=>chart(id,rows,keys,id==='mapChart'?['mAP50','mAP50–95']:id==='prChart'?['Precision','Recall']:['Train','Validation']));
   const keys=['epoch','box_train','box_val','cls_train','cls_val','dfl_train','dfl_val','precision','recall','map50','map50_95'];
   el('trainingRows').innerHTML=rows.length?rows.map(r=>`<tr>${keys.map(k=>`<td>${numeric(r[k])?(k==='epoch'?r[k]:r[k].toFixed(4)):'—'}</td>`).join('')}</tr>`).join(''):'<tr><td colspan="11">ยังไม่มีข้อมูลราย epoch จากการเทรนจริง</td></tr>';
   const csv=document.querySelector('a[href="/api/training-csv"]');csv.hidden=!data.csv_available;
   comparison(data.comparison);
  }catch(error){
   el('trainingStatus').textContent='อ่านสถานะการเทรนไม่ได้ จะลองใหม่ในรอบถัดไป';el('trainingError').textContent=error.message;
  }finally{loading=false}
 }
 document.querySelector('[data-tab="training"]').addEventListener('click',refreshTraining);
 refreshTraining();setInterval(refreshTraining,15000);
})();
