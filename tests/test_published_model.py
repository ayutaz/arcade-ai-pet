import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
from arcade_ai_pet.frozen import read_layers
from arcade_ai_pet.training import scores,decisions

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models/kws-v2'

class PublishedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata=json.loads((MODEL/'metadata.json').read_text())
        cls.layers,_=read_layers(MODEL/'weights.npz')
        cls.temp=tempfile.TemporaryDirectory();out=Path(cls.temp.name)/'model.dylib'
        command=['clang','-std=c11','-O2','-shared','-fPIC','-I'+str(ROOT/'native'),'-I'+str(MODEL),
                 str(ROOT/'native/features.c'),str(ROOT/'native/inference.c'),'-lm','-o',str(out)]
        result=subprocess.run(command,capture_output=True,text=True)
        if result.returncode:raise AssertionError(result.stderr)
        cls.lib=ctypes.CDLL(str(out));cls.lib.pet_predict.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t]
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_published_model_identity_and_integer_c_match_golden_features(self):
        self.assertEqual(hashlib.sha256((MODEL/'weights.npz').read_bytes()).hexdigest(),self.metadata['weights_sha256'])
        with np.load(MODEL/'conformance.npz',allow_pickle=False) as data:
            x=data['features'];expected=data['logits']
        np.testing.assert_array_equal(expected,scores(x,self.layers))
        self.assertLessEqual(self.metadata['weights_bytes'],24576)
        self.assertFalse(self.metadata['device_tested'])
        for vector,logits in zip(x,expected,strict=True):
            actual=np.zeros(7,dtype=np.int32)
            result=self.lib.pet_predict(vector.ctypes.data,260,actual.ctypes.data,7)
            np.testing.assert_array_equal(actual,logits)
            self.assertEqual(result,int(decisions(logits[None,:],0)[0]))

    def test_heldout_evidence_is_explicitly_synthetic_and_passes_fixed_targets(self):
        report=json.loads((MODEL/'evaluation.json').read_text())
        self.assertTrue(report['accepted_synthetic']);self.assertGreaterEqual(report['macro_f1'],.9)
        self.assertLessEqual(report['false_activation_rate'],.05);self.assertEqual(report['source_overlap'],0)
        self.assertFalse(report['real_voice_tested']);self.assertFalse(report['device_tested'])
