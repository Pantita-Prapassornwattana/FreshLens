"""Local pretrained produce inference and deterministic box suppression. No training."""
from pathlib import Path
import time
from tests.evaluate_pretrained import iou

ROOT=Path(__file__).resolve().parent

EXPANSION_MAP={
    'Apple':'apple','Banana':'banana','Beetroot':'beet','Bitter_Gourd':'bitter gourd',
    'Bottle_Gourd':'bottle gourd','Cabbage':'cabbage','Capsicum':'bell pepper',
    'Carrot':'carrot','Cauliflower':'cauliflower','Cherry':'cherry','Chilli':'hot pepper',
    'Coconut':'coconut','Cucumber':'cucumber','EggPlant':'eggplant','Ginger':'ginger',
    'Grape':'grape','Green_Orange':'orange','Kiwi':'kiwi','Maize':'corn',
    'Mango':'mango','Melon':'melon','Okra':'okra','Onion':'onion','Orange':'orange',
    'Peach':'peach','Pear':'pear','Peas':'pea','Pineapple':'pineapple',
    'Pomegranate':'pomegranate','Potato':'potato','Radish':'radish',
    'Strawberry':'strawberry','Tomato':'tomato','Turnip':'turnip','Watermelon':'watermelon'}

def uniform_image(image):
    """A nearly constant canvas has no visible object boundaries to identify."""
    return all(high-low<=2 for low,high in image.getextrema())

def suppress(boxes,threshold=.45,agnostic=False):
    kept=[]
    for box in sorted(boxes,key=lambda b:b['confidence'],reverse=True):
        if not any((agnostic or box['class']==other['class']) and iou(box['box'],other['box'])>threshold for other in kept):
            kept.append(box)
    return kept

class ProduceDetector:
    def __init__(self,config):
        from ultralytics import YOLO
        self.config=config
        loaded={}
        self.models=[]
        for member in config['members']:
            weights=member['weights']
            if weights not in loaded:loaded[weights]=YOLO(str(ROOT/weights))
            self.models.append(loaded[weights])

    def predict(self,image):
        started=time.perf_counter()
        if self.config.get('reject_uniform_images') and uniform_image(image):
            return [],(time.perf_counter()-started)*1000
        boxes=[]
        for index,(model,member) in enumerate(zip(self.models,self.config['members'])):
            mapping=member['class_map']
            allowed=set(member.get('active_classes',mapping.values()))
            accepted=[int(i) for i,n in model.names.items() if n in mapping and mapping[n] in allowed]
            if not accepted:continue
            regions=[(image,0,0)]
            if member.get('tiles'):
                w,h=image.size
                tw,th=round(w*.6),round(h*.6)
                regions += [(image.crop((x,y,x+tw,y+th)),x,y) for y in (0,h-th) for x in (0,w-tw)]
            existing=list(boxes)
            member_boxes=[]
            for crop,offset_x,offset_y in regions:
                result=model.predict(crop,conf=member.get('confidence_threshold',.25),
                    iou=.45,imgsz=member.get('imgsz',640),augment=member.get('augment',False),classes=accepted,device=0,verbose=False)[0]
                for b in result.boxes:
                    name=mapping[result.names[int(b.cls.item())]]
                    score=float(b.conf.item())
                    if score<member.get('class_thresholds',{}).get(name,member.get('confidence_threshold',.25)):continue
                    coords=[float(v) for v in b.xyxy[0].tolist()]
                    coords=[coords[0]+offset_x,coords[1]+offset_y,coords[2]+offset_x,coords[3]+offset_y]
                    member_boxes.append({'class':name,
                        'confidence':round(score,4),
                        'box':[round(v,1) for v in coords],'source':member['id']})
            if self.config.get('supplement_only') and index>0:
                member_boxes=[b for b in member_boxes if not any(iou(b['box'],old['box'])>self.config.get('supplement_overlap',.3) for old in existing)]
            boxes.extend(member_boxes)
        boxes=suppress(boxes,self.config.get('merge_iou',.45),self.config.get('agnostic_merge',False))
        return boxes,(time.perf_counter()-started)*1000

    def close(self):
        for model in self.models:model.cpu()
