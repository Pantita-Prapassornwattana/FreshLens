"""Live localhost contract checks using synthetic controls only; no training."""
import base64
import hashlib
import http.client
import io
import json
import math
import unittest
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def history_digest():
    path = ROOT / 'artifacts/history.json'
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


class LocalAPIContractTests(unittest.TestCase):
    records = []

    @classmethod
    def tearDownClass(cls):
        path = ROOT / 'artifacts/api_checks.json'
        path.write_text(json.dumps({'created_at': datetime.now(timezone.utc).isoformat(),
            'target': 'http://127.0.0.1:8000', 'fixtures': 'synthetic only; no final-test images rerun',
            'all_passed': all(row['passed'] for row in cls.records) and len(cls.records) == 4,
            'checks': cls.records}, ensure_ascii=False, indent=2), encoding='utf-8')

    def post(self, body, headers):
        connection = http.client.HTTPConnection('127.0.0.1', 8000, timeout=120)
        try:
            connection.request('POST', '/api/predict', body=body, headers=headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def row(self, name, actual, expected):
        record = {'name': name, 'http_status': actual, 'expected_status': expected, 'passed': False}
        self.records.append(record)
        self.assertEqual(actual, expected)
        return record

    def test_malformed_image_returns_400(self):
        status, payload = self.post(b'this is not an image', {'Content-Type': 'image/jpeg', 'X-Evaluation': 'true'})
        record = self.row('malformed image rejected', status, 400)
        self.assertIsInstance(payload['error'], str)
        record['passed'] = True

    def test_cross_origin_request_returns_403(self):
        status, payload = self.post(b'invalid', {'Content-Type': 'image/jpeg', 'Origin': 'https://example.invalid'})
        record = self.row('cross origin blocked before inference', status, 403)
        self.assertIsInstance(payload['error'], str)
        record['passed'] = True

    def test_oversize_content_length_rejected_before_reading_body(self):
        connection = http.client.HTTPConnection('127.0.0.1', 8000, timeout=15)
        try:
            connection.putrequest('POST', '/api/predict')
            connection.putheader('Content-Length', str(12 * 1024 * 1024 + 1))
            connection.putheader('Content-Type', 'image/jpeg')
            connection.endheaders()  # Deliberately send no body; reject based on header.
            response = connection.getresponse()
            payload = json.loads(response.read())
            record = self.row('oversized request rejected without receiving body', response.status, 413)
            self.assertIsInstance(payload['error'], str)
            record['passed'] = True
        finally:
            connection.close()

    def test_negative_png_schema_and_evaluation_history(self):
        before = history_digest()
        fixture = ROOT / 'tests/fixtures/negative_controls/white.png'
        status, payload = self.post(fixture.read_bytes(), {'Content-Type': 'image/png', 'X-Evaluation': 'true'})
        record = self.row('negative PNG inference schema and no evaluation history writes', status, 200)
        expected = {'id', 'time', 'run_id', 'latency_ms', 'detections', 'image'}
        self.assertTrue(expected.issubset(payload))
        self.assertIsInstance(payload['id'], str)
        self.assertIsInstance(payload['time'], str)
        selected = json.loads((ROOT / 'artifacts/selected_model.json').read_text(encoding='utf-8'))
        self.assertEqual(payload['run_id'], selected['run_id'])
        self.assertTrue(math.isfinite(payload['latency_ms']) and payload['latency_ms'] >= 0)
        self.assertEqual(payload['detections'], [])
        self.assertTrue(payload['image'].startswith('data:image/jpeg;base64,'))
        with Image.open(io.BytesIO(base64.b64decode(payload['image'].split(',', 1)[1], validate=True))) as result:
            self.assertEqual(result.size, (640, 480))
            self.assertEqual(result.format, 'JPEG')
        self.assertEqual(history_digest(), before)
        record.update({'detections': 0, 'history_unchanged': True, 'schema_valid': True,
                       'annotated_image_decodes': True, 'passed': True})


if __name__ == '__main__':
    unittest.main(verbosity=2)
