import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import prepare

class AuditTest(unittest.TestCase):
    def test_cross_split_duplicates_and_invalid_labels_are_excluded(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'.tmp') as tmp:
            root=Path(tmp); source=root/'raw';source.mkdir()
            (source/'data.yaml').write_text('names: [apple]\n',encoding='utf-8')
            for folder,color in [('train','red'),('valid','green'),('test','blue')]:
                images=source/folder/'images';labels=source/folder/'labels'
                images.mkdir(parents=True);labels.mkdir()
                Image.new('RGB',(20,20),color).save(images/f'{folder}.png')
                (labels/f'{folder}.txt').write_text('0 0.5 0.5 0.4 0.4\n')
            Image.new('RGB',(20,20),'blue').save(source/'train/images/copied.png')
            (source/'train/labels/copied.txt').write_text('0 0.5 0.5 0.4 0.4\n')
            Image.new('RGB',(20,20),'yellow').save(source/'train/images/bad.png')
            (source/'train/labels/bad.txt').write_text('0 0.5 0.5 -0.4 0.4\n')
            with patch.object(prepare,'ROOT',root):
                prepare.prepare(source)
            result=json.loads((root/'artifacts/dataset_audit.json').read_text())
            self.assertEqual(result['counts'],{'train':1,'val':1,'test':1})
            self.assertEqual(len(result['excluded']),2)
            paths=[set((root/f'data/prepared/{s}.txt').read_text().splitlines()) for s in ['train','val','test']]
            self.assertFalse(paths[0]&paths[1] or paths[0]&paths[2] or paths[1]&paths[2])

if __name__=='__main__':unittest.main()
