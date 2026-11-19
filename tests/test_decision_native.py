"""新261特徴・5閾値の実C/分割bankを独立したPython演算と照合する。"""
import ctypes as C
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import decision_assets as assets, decision_native as native
from arcade_ai_pet.duration_frontend import features
from arcade_ai_pet.channel_robustness import condition
from arcade_ai_pet.channel_training import level
from arcade_ai_pet.training import scores
from arcade_ai_pet.class_rejection import decisions
from test_decision_assets import model


class DecisionNativeTests(unittest.TestCase):
    def fixture(self,root):
        layers=model(root/'weights.npz');thresholds=[10,20,30,40,50]
        assets.export_assets(root/'weights.npz',thresholds,root/'assets')
        return layers,np.asarray(thresholds,dtype=np.int32)

    def test_forged_header_is_refused_before_compiling_even_if_supplied_checksum_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();self.fixture(root);file=root/'assets/decision_model.h';file.write_text('#include "/tmp/never-compile"')
            path=root/'assets/decision-metadata.json';meta=json.loads(path.read_text());meta['header_sha256']['decision_model.h']=hashlib.sha256(file.read_bytes()).hexdigest();path.write_text(json.dumps(meta))
            with patch('arcade_ai_pet.decision_native.subprocess.run',side_effect=AssertionError('compiler reached')):
                with self.assertRaises(ValueError):native.DecisionNativeModel(root/'assets')

    def test_features_and_banked_preparation_match_python_without_changing_caller_pcm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();self.fixture(root);rng=np.random.default_rng(39);t=np.arange(32000)/16000
            quiet=rng.normal(0,3,32000);quiet[6400:16000]+=600*np.sin(2*np.pi*720*t[:9600])
            signals=[np.zeros(32000,dtype=np.int16),np.full(32000,1600,dtype=np.int16),np.tile(np.array([-32768,32767],dtype=np.int16),16000),quiet.astype(np.int16),rng.integers(-1000,1001,32000,dtype=np.int16)]
            with native.DecisionNativeModel(root/'assets') as engine:
                self.assertLessEqual(engine.scratch_bytes,24576)
                for pcm in signals:
                    before=pcm.copy();prepared=level(condition(pcm,'hp160'));expected=features(prepared,context=True)
                    np.testing.assert_array_equal(engine.prepare(pcm),prepared)
                    actual=engine.extract(pcm);banked=engine.extract(pcm,banked=True)
                    np.testing.assert_array_equal(actual,banked);self.assertEqual(actual.shape,(261,))
                    self.assertLess(float(abs(actual-expected).max()),.001)
                    self.assertAlmostEqual(float(actual[-1]),float(expected[-1]),places=6)
                    np.testing.assert_array_equal(pcm,before)

    def test_integer_scores_and_class_rejection_match_python_including_extreme_margins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();layers,thresholds=self.fixture(root)
            with native.DecisionNativeModel(root/'assets') as engine:
                values=np.random.default_rng(9).uniform(-16,16,(40,261)).astype(np.float32)
                values=np.vstack((values,np.zeros((1,261),dtype=np.float32),np.full((1,261),.5/1024,dtype=np.float32)))
                for row in values:
                    expected=scores(row[None,:],layers);label,logits=engine.predict(row)
                    np.testing.assert_array_equal(logits,expected[0]);self.assertEqual(label,int(decisions(expected,thresholds)[0]))
                raw=np.full(7,-(2**31),dtype=np.int32);raw[0]=2**31-1
                self.assertEqual(engine.lib.pet_decision_classify(raw.ctypes.data,7,thresholds.ctypes.data,5),0)
                raw[:]=0;raw[5]=1
                self.assertEqual(engine.lib.pet_decision_classify(raw.ctypes.data,7,thresholds.ctypes.data,5),5)

    def test_native_api_rejects_wrong_shapes_null_workspaces_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();self.fixture(root)
            with native.DecisionNativeModel(root/'assets') as engine:
                pcm=np.zeros(32000,dtype=np.int16);out=np.zeros(261,dtype=np.float32);work=C.create_string_buffer(engine.scratch_bytes+16)
                for samples,count,bytes_ in ((31999,261,engine.scratch_bytes),(32000,260,engine.scratch_bytes),(32000,261,engine.scratch_bytes-1)):
                    self.assertEqual(engine.lib.pet_decision_features(pcm.ctypes.data,samples,out.ctypes.data,count,work,bytes_),-1)
                self.assertEqual(engine.lib.pet_decision_features(None,32000,out.ctypes.data,261,work,engine.scratch_bytes),-1)
                self.assertEqual(engine.lib.pet_decision_features(pcm.ctypes.data,32000,out.ctypes.data,261,C.addressof(work)+1,engine.scratch_bytes),-1)
                logits=np.zeros(7,dtype=np.int32);out[0]=np.nan
                self.assertEqual(engine.lib.pet_decision_predict(out.ctypes.data,261,logits.ctypes.data,7),-1)
                self.assertEqual(engine.lib.pet_decision_predict(out.ctypes.data,260,logits.ctypes.data,7),-1)
                thresholds=np.array([-1,0,0,0,0],dtype=np.int32)
                self.assertEqual(engine.lib.pet_decision_classify(logits.ctypes.data,7,thresholds.ctypes.data,5),-1)


if __name__=='__main__':unittest.main()
