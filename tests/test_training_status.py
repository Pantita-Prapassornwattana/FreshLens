import json
import tempfile
import unittest
from pathlib import Path
from training_status import training_state, training_csv

class TrainingStatusTests(unittest.TestCase):
    def test_no_run_returns_explicit_empty_state(self):
        with tempfile.TemporaryDirectory() as directory:
            state=training_state(directory)
            self.assertEqual(state['status'],'not_started')
            self.assertEqual(state['rows'],[])
            self.assertFalse(state['csv_available'])

    def test_csv_reads_real_rows_preserves_missing_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'artifacts').mkdir();(root/'runs/run-a').mkdir(parents=True)
            (root/'artifacts/experiments.json').write_text(json.dumps([{'run_id':'run-a','status':'completed','requested_epochs':2}]))
            (root/'runs/run-a/results.csv').write_text(' epoch, train/box_loss, metrics/recall(B)\n1, 2.5, 0.3\n2, 2.0, nan\n3, , \n')
            state=training_state(root)
            self.assertEqual(state['completed_epochs'],2)
            self.assertEqual(state['rows'][0]['recall'],0.3)
            self.assertIsNone(state['latest']['recall'])
            self.assertIsNone(state['latest']['map50_95'])
            self.assertIn('2,2.0',training_csv(state).decode('utf-8-sig'))

    def test_record_cannot_read_csv_outside_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'artifacts').mkdir()
            (root/'artifacts/experiments.json').write_text(json.dumps([{'run_id':'../artifacts'}]))
            (root/'artifacts/results.csv').write_text('epoch,train/box_loss\n1,1\n')
            self.assertEqual(training_state(root)['rows'],[])
