import unittest
from produce_detector import suppress,EXPANSION_MAP,uniform_image
from PIL import Image,ImageDraw

class MergeTests(unittest.TestCase):
    def test_uniform_canvas_guard_keeps_visible_objects(self):
        canvas=Image.new('RGB',(64,64),'white')
        self.assertTrue(uniform_image(canvas))
        ImageDraw.Draw(canvas).rectangle((20,20,30,30),fill='green')
        self.assertFalse(uniform_image(canvas))
    def test_duplicate_boxes_keep_highest_confidence(self):
        low={'class':'cabbage','confidence':.4,'box':[0,0,20,20]}
        high={'class':'cabbage','confidence':.9,'box':[1,1,21,21]}
        self.assertEqual(suppress([low,high]),[high])

    def test_adjacent_objects_preserved_and_conflicting_labels_optional(self):
        a={'class':'apple','confidence':.9,'box':[0,0,20,20]}
        b={'class':'banana','confidence':.8,'box':[0,0,20,20]}
        c={'class':'apple','confidence':.7,'box':[25,0,45,20]}
        self.assertEqual(len(suppress([a,b,c])),3)
        self.assertEqual(suppress([a,b,c],agnostic=True),[a,c])

    def test_mapping_preserves_vegetable_types(self):
        self.assertEqual(EXPANSION_MAP['Cabbage'],'cabbage')
        self.assertEqual(EXPANSION_MAP['Beetroot'],'beet')
        self.assertEqual(EXPANSION_MAP['Okra'],'okra')
        self.assertNotIn('beans',EXPANSION_MAP.values())

if __name__=='__main__':unittest.main()
