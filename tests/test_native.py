import ctypes
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
from arcade_ai_pet.features import features
from arcade_ai_pet.frozen import read_layers
from arcade_ai_pet.quantize import predict
from arcade_ai_pet.quantize import quantize_layer
from arcade_ai_pet.training import scores
from arcade_ai_pet.native import export_assets
from test_frozen import dataset,checkpoint
from arcade_ai_pet.frozen import freeze_model

ROOT=Path(__file__).resolve().parents[1]

class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.base=Path(cls.temp.name)
        dataset(cls.base/'data','native-development');checkpoint(cls.base/'checkpoint')
        rng=np.random.default_rng(147)
        layers=[quantize_layer(np.eye(7,260)+rng.normal(0,.01,(7,260)),rng.normal(0,.01,7)),
                quantize_layer(np.eye(7)+rng.normal(0,.01,(7,7)),rng.normal(0,.01,7))]
        np.savez(cls.base/'checkpoint/weights.npz',**{f'{field}{i}':layer[field] for i,layer in enumerate(layers) for field in ('weight','bias','shift')})
        freeze_model(cls.base/'checkpoint',cls.base/'data',cls.base/'frozen')
        export_assets(cls.base/'frozen',cls.base/'assets')
        result=subprocess.run(['clang','-std=c11','-O2','-shared','-fPIC','-I'+str(ROOT/'native'),
            '-I'+str(cls.base/'assets'),str(ROOT/'native/features.c'),str(ROOT/'native/inference.c'),
            '-lm','-o',str(cls.base/'native.dylib')],capture_output=True,text=True)
        if result.returncode:raise AssertionError(result.stderr)
        cls.lib=ctypes.CDLL(str(cls.base/'native.dylib'))
        cls.lib.pet_feature_scratch_size.restype=ctypes.c_size_t
        cls.lib.pet_features.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t]
        cls.lib.pet_predict.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t]
        cls.layers,_=read_layers(cls.base/'frozen/weights.npz')
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_c_mfcc_matches_python_for_silence_tones_voice_like_and_noise(self):
        rng=np.random.default_rng(73);signals=[np.zeros(32000,dtype=np.int16)]
        t=np.arange(32000)/16000
        signals.append((5000*np.sin(2*np.pi*650*t)).astype(np.int16))
        voice=np.zeros(32000);voice[4800:20800]=4000*np.sin(2*np.pi*(220*t[:16000]+150*t[:16000]**2))
        signals.append(voice.astype(np.int16));signals.append(rng.integers(-32000,32000,32000,dtype=np.int16))
        size=self.lib.pet_feature_scratch_size();self.assertLessEqual(size,24576)
        for pcm in signals:
            actual=np.zeros(260,dtype=np.float32);scratch=ctypes.create_string_buffer(size)
            self.assertEqual(self.lib.pet_features(pcm.ctypes.data,len(pcm),actual.ctypes.data,260,scratch,size),1)
            self.assertLess(float(abs(actual-features(pcm)).max()),.01)

    def test_c_integer_scores_match_python_including_ties_and_bounds(self):
        rng=np.random.default_rng(98)
        vectors=[np.zeros(260,dtype=np.float32),np.full(260,16,dtype=np.float32),np.full(260,-16,dtype=np.float32)]
        vectors.extend(rng.normal(0,2,(50,260)).astype(np.float32))
        for vector in vectors:
            actual=np.zeros(7,dtype=np.int32)
            decision=self.lib.pet_predict(vector.ctypes.data,260,actual.ctypes.data,7)
            np.testing.assert_array_equal(actual,scores(vector[None,:],self.layers)[0])
            self.assertEqual(decision,predict(vector,self.layers))

    def test_native_apis_reject_bad_lengths_null_and_nonfinite_inputs(self):
        pcm=np.zeros(32000,dtype=np.int16);output=np.zeros(260,dtype=np.float32)
        size=self.lib.pet_feature_scratch_size();scratch=ctypes.create_string_buffer(size)
        for count,capacity,work in ((31999,260,size),(32000,259,size),(32000,260,size-1)):
            self.assertEqual(self.lib.pet_features(pcm.ctypes.data,count,output.ctypes.data,capacity,scratch,work),-1)
        self.assertEqual(self.lib.pet_features(None,32000,output.ctypes.data,260,scratch,size),-1)
        logits=np.zeros(7,dtype=np.int32);output[0]=np.nan
        self.assertEqual(self.lib.pet_predict(output.ctypes.data,260,logits.ctypes.data,7),-1)

    def test_export_is_deterministic_and_refuses_foreign_output(self):
        other=self.base/'other-assets';export_assets(self.base/'frozen',other)
        for name in ('feature_tables.h','pet_model.h'):
            self.assertEqual((other/name).read_bytes(),(self.base/'assets'/name).read_bytes())
        with self.assertRaises(ValueError):export_assets(self.base/'frozen',other)

    def test_model_metadata_cannot_inject_c_into_exported_constants(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);shutil.copytree(self.base/'frozen',base/'frozen')
            path=base/'frozen/frozen.json';metadata=json.loads(path.read_text())
            metadata['threshold_q10']='0}; int unwanted; enum {X=0'
            path.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):export_assets(base/'frozen',base/'output')
            self.assertFalse((base/'output').exists())
