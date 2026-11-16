import ctypes
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
from arcade_ai_pet.behavior_native import export_behavior,load_behavior
from arcade_ai_pet.behavior import World,observation
from arcade_ai_pet.quantize import predict

ROOT=Path(__file__).resolve().parents[1]

class BehaviorNativeTests(unittest.TestCase):
    def test_bounded_export_and_c_integer_decisions_match(self):
        rng=np.random.default_rng(93)
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source=base/'weights.npz'
            np.savez(source,weight0=rng.integers(-12,12,(32,6),dtype=np.int8),bias0=np.zeros(32,dtype=np.int32),shift0=3,
                     weight1=rng.integers(-12,12,(4,32),dtype=np.int8),bias1=np.zeros(4,dtype=np.int32),shift1=4)
            export_behavior(source,base/'export')
            layers=load_behavior(source)
            output=base/'policy.so'
            subprocess.run(['clang','-O2','-shared','-fPIC','-std=c11','-I'+str(base/'export'),str(ROOT/'native/behavior.c'),'-o',str(output)],check=True,capture_output=True)
            lib=ctypes.CDLL(str(output));lib.pet_behavior.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
            for seed in range(64):
                world=World(seed);q=np.rint(observation(world)*1024).astype(np.int32);scores=np.zeros(4,dtype=np.int32)
                self.assertEqual(lib.pet_behavior(q.ctypes.data,6,scores.ctypes.data),predict(observation(world),layers))
            bad=np.full(6,1025,dtype=np.int32)
            self.assertEqual(lib.pet_behavior(bad.ctypes.data,6,scores.ctypes.data),-1)
            with self.assertRaises(ValueError):export_behavior(source,base/'export')
            np.savez(base/'bad.npz',weight0=np.zeros((1000,6),dtype=np.int8))
            with self.assertRaises(ValueError):load_behavior(base/'bad.npz')

    def test_published_policy_matches_c_for_all_heldout_rollout_steps(self):
        import hashlib
        from arcade_ai_pet.behavior import benchmark,random_action
        from arcade_ai_pet.quantize import predict
        model=ROOT/'models/behavior-v1'
        report=json.loads((model/'evaluation.json').read_text())
        self.assertEqual(hashlib.sha256((model/'weights.npz').read_bytes()).hexdigest(),report['weights_sha256'])
        self.assertTrue(report['accepted']);self.assertFalse(report['device_tested'])
        self.assertFalse(set(report['train_seeds'])&set(report['test_seeds']))
        layers=load_behavior(model/'weights.npz')
        with tempfile.TemporaryDirectory() as tmp:
            library=Path(tmp)/'policy.so'
            subprocess.run(['clang','-O2','-shared','-fPIC','-std=c11','-I'+str(model),str(ROOT/'native/behavior.c'),'-o',str(library)],check=True,capture_output=True)
            lib=ctypes.CDLL(str(library));lib.pet_behavior.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
            scores=np.zeros(4,dtype=np.int32)
            def policy(world):
                vector=observation(world);q=np.rint(vector*1024).astype(np.int32)
                actual=lib.pet_behavior(q.ctypes.data,6,scores.ctypes.data)
                self.assertEqual(actual,predict(vector,layers))
                return actual
            actual=benchmark(policy,seeds=report['test_seeds'],steps=120)
            self.assertEqual(actual,report['test']['learned'])
