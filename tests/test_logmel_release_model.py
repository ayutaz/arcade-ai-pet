"""Released small numeric model must reproduce the frozen runtime and evaluation."""
from pathlib import Path
import hashlib,json,unittest
import numpy as np
from arcade_ai_pet.dscnn_assets import DscnnNativeAssets
from arcade_ai_pet.dscnn_freeze import verify
ROOT=Path(__file__).resolve().parents[1];MODEL=ROOT/'models/kws-logmel-v1'
class LogmelReleaseModelTests(unittest.TestCase):
 def test_numeric_export_reproduces_frozen_model_and_silence_in_both_layouts(self):
  meta,layers=verify(MODEL)
  self.assertEqual(meta['weights_sha256'],'194c5fceaeb9e51f83d8637033680a779c50c614becc3ff7bc37339ec31d27a8');self.assertEqual(meta['numeric_bytes'],23411)
  self.assertEqual(meta['thresholds_q10'],[3505,1220,3063,552,1052]);self.assertEqual(len(layers),6)
  with DscnnNativeAssets(MODEL) as native:
   raw=np.zeros(32000,np.int16);a=native.extract(raw);b=native.extract(raw,banked=True)
   np.testing.assert_array_equal(a,b);self.assertEqual(native.predict(a)[0],6)
 def test_final_evidence_license_and_small_asset_boundary_are_preserved(self):
  r=json.loads((MODEL/'evaluation.json').read_text());self.assertTrue(r['accepted_synthetic']);self.assertTrue(r['native_conformance']['matched']);self.assertEqual(r['native_conformance']['rows'],3448)
  self.assertEqual(r['model_binding']['frozen_sha256'],hashlib.sha256((MODEL/'decision-frozen.json').read_bytes()).hexdigest());self.assertFalse(r['real_voice_tested'])
  self.assertEqual(len(r['profiles']),8)
  for row in r['profiles']:
   self.assertGreaterEqual(row['macro_f1'],.9);self.assertLessEqual(row['false_activation_rate'],.05);self.assertGreaterEqual(row['command_recall'],.8)
  license=(MODEL/'LICENSE.txt').read_text()
  for name in ('CC BY 3.0','Mei','M001','Irodori','Semantic-DACVAE','Apache-2.0'):self.assertIn(name,license)
  self.assertLess((MODEL/'weights.npz').stat().st_size,65536)
  self.assertFalse(any(p.suffix in ('.wav','.pcm','.safetensors','.bin','.labapp') for p in MODEL.iterdir()))
