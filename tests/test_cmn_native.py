"""CMN専用形式と実Cを、旧261特徴の意味から分離して照合する。"""
import ctypes as C
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import cmn_assets, cmn_native
from arcade_ai_pet import cmn_frontend, decision_assets, decision_native
from arcade_ai_pet.channel_robustness import condition
from arcade_ai_pet.channel_training import level
from arcade_ai_pet.class_rejection import decisions
from arcade_ai_pet.training import scores
from test_decision_assets import model


class CmnNativeTests(unittest.TestCase):
    def fixture(self, root):
        layers = model(root / 'weights.npz')
        table = [10, 20, 30, 40, 50]
        meta = cmn_assets.export_assets(root / 'weights.npz', table, root / 'assets')
        return layers, table, meta

    def test_explicit_cmn_contract_cannot_enter_either_other_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            _, _, meta = self.fixture(root)
            self.assertEqual(meta['runtime'], 'context-duration-cmn-class-v1')
            self.assertEqual(meta['input_contract']['frontend'], 'context-duration-cmn')
            self.assertEqual(meta['input_contract']['cmn'], 'post-bin-mean-20-clip16')
            self.assertFalse(meta['frozen'])
            decision_assets.export_assets(root / 'weights.npz', [0]*5, root / 'legacy')
            with patch('arcade_ai_pet.decision_native.subprocess.run', side_effect=AssertionError('compiler reached')):
                with self.assertRaises(ValueError):
                    decision_native.DecisionNativeModel(root / 'assets')
                with self.assertRaises(ValueError):
                    cmn_native.CmnNativeModel(root / 'legacy')

    def test_forged_header_refused_even_with_matching_user_supplied_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            self.fixture(root)
            header = root / 'assets/decision_model.h'
            header.write_text('#include "/tmp/never-compile"')
            path = root / 'assets/decision-metadata.json'
            meta = json.loads(path.read_text())
            meta['header_sha256'][header.name] = hashlib.sha256(header.read_bytes()).hexdigest()
            path.write_text(json.dumps(meta))
            with patch('arcade_ai_pet.decision_native.subprocess.run', side_effect=AssertionError('compiler reached')):
                with self.assertRaises(ValueError):
                    cmn_native.CmnNativeModel(root / 'assets')

    def test_inplace_center_is_exact_and_preserves_energy_and_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            self.fixture(root)
            rng = np.random.default_rng(71)
            rows = rng.uniform(-16,16,(32,261)).astype(np.float32)
            rows[:,-1] = rng.uniform(0,4,32)
            rows[0,:240] = 0
            rows[0,0] = 16  # mean subtraction and final clipping both matter
            rows[1,:240] = -16
            rows[1,0] = 16
            with cmn_native.CmnNativeModel(root / 'assets') as engine:
                for row in rows:
                    expected = cmn_frontend.center(row)
                    actual = row.copy()
                    self.assertEqual(engine.lib.pet_cmn_center(actual.ctypes.data,261),1)
                    np.testing.assert_array_equal(actual,expected)
                    np.testing.assert_array_equal(actual[240:],row[240:])

    def test_pcm_features_and_banks_match_python_without_mutating_caller(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            self.fixture(root)
            rng = np.random.default_rng(39)
            pcm = rng.integers(-3,4,32000,dtype=np.int16)
            pcm[8000:17000] += (900*np.sin(np.arange(9000)*.29)).astype(np.int16)
            signals = [pcm,np.zeros(32000,dtype=np.int16),np.full(32000,1200,dtype=np.int16),rng.integers(-1000,1001,32000,dtype=np.int16)]
            with cmn_native.CmnNativeModel(root / 'assets') as engine:
                self.assertEqual(engine.scratch_bytes,5680)
                self.assertIn('cmn_features.c',engine.source_sha256)
                for signal in signals:
                    original = signal.copy()
                    expected = cmn_frontend.features(level(condition(signal,'hp160')))
                    actual = engine.extract(signal)
                    np.testing.assert_array_equal(actual,engine.extract(signal,banked=True))
                    self.assertLess(float(abs(expected-actual).max()),.001)
                    np.testing.assert_array_equal(original,signal)

    def test_integer_scores_and_rejection_use_the_selected_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            layers, table, _ = self.fixture(root)
            values = np.random.default_rng(9).uniform(-16,16,(40,261)).astype(np.float32)
            values[:,-1] = 0
            with cmn_native.CmnNativeModel(root / 'assets') as engine:
                for row in cmn_frontend.center(values):
                    expected = scores(row[None,:],layers)
                    label, logits = engine.predict(row)
                    np.testing.assert_array_equal(logits,expected[0])
                    self.assertEqual(label,int(decisions(expected,np.asarray(table,dtype=np.int32))[0]))

    def test_invalid_center_and_feature_buffers_are_refused_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            self.fixture(root)
            with cmn_native.CmnNativeModel(root / 'assets') as engine:
                for index, value in ((0,np.nan),(0,np.inf),(0,17),(260,-1),(260,5)):
                    row = np.zeros(261,dtype=np.float32)
                    row[index] = value
                    before = row.tobytes()
                    self.assertEqual(engine.lib.pet_cmn_center(row.ctypes.data,261),-1)
                    self.assertEqual(row.tobytes(),before)
                self.assertEqual(engine.lib.pet_cmn_center(None,261),-1)
                pcm = np.zeros(32000,dtype=np.int16)
                out = np.full(261,8,dtype=np.float32)
                work = C.create_string_buffer(engine.scratch_bytes+16)
                for samples,count,size in ((31999,261,engine.scratch_bytes),(32000,260,engine.scratch_bytes),(32000,261,engine.scratch_bytes-1)):
                    self.assertEqual(engine.lib.pet_cmn_features(pcm.ctypes.data,samples,out.ctypes.data,count,work,size),-1)
                    np.testing.assert_array_equal(out,8)
                self.assertEqual(engine.lib.pet_cmn_features(pcm.ctypes.data,32000,out.ctypes.data,261,C.addressof(work)+1,engine.scratch_bytes),-1)
                banks = (C.c_void_p*8)(*([pcm.ctypes.data]*7+[None]))
                self.assertEqual(engine.lib.pet_cmn_features_banked(banks,32000,out.ctypes.data,261,work,engine.scratch_bytes),-1)
                np.testing.assert_array_equal(out,8)


if __name__ == '__main__':
    unittest.main()
