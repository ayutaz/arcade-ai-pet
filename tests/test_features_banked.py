import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
from arcade_ai_pet.features import features
ROOT=Path(__file__).resolve().parents[1]

class BankedTests(unittest.TestCase):
    def test_eight_pcm_banks_match_contiguous_features_without_a_pcm_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            library=Path(tmp)/'model.so'
            subprocess.run(['clang','-O2','-shared','-fPIC','-std=c11','-I'+str(ROOT/'native'),'-I'+str(ROOT/'models/kws-v2'),str(ROOT/'native/features.c'),'-lm','-o',str(library)],check=True,capture_output=True)
            lib=ctypes.CDLL(str(library));lib.pet_feature_scratch_size.restype=ctypes.c_size_t
            function=lib.pet_features_banked
            function.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t]
            size=lib.pet_feature_scratch_size();workspace=ctypes.create_string_buffer(size)
            rng=np.random.default_rng(741)
            for pcm in (np.zeros(32000,dtype=np.int16),rng.integers(-3000,3000,32000,dtype=np.int16)):
                banks=[np.ascontiguousarray(pcm[i:i+4096]) for i in range(0,32000,4096)]
                pointers=(ctypes.c_void_p*8)(*(x.ctypes.data for x in banks));actual=np.zeros(260,dtype=np.float32)
                self.assertEqual(function(pointers,32000,actual.ctypes.data,260,workspace,size),1)
                self.assertLess(float(abs(actual-features(pcm)).max()),.01)
                pointers[3]=None
                self.assertEqual(function(pointers,32000,actual.ctypes.data,260,workspace,size),-1)
