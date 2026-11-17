import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import channel_freeze as freeze
from arcade_ai_pet.channel_robustness import PROFILES
from test_frozen import checkpoint


class ChannelFreezeTests(unittest.TestCase):
    def fixture(self,root):
        model=root/'model';checkpoint(model);dev=root/'dev';dev.mkdir()
        row=dict(id=hashlib.sha256(b'dev/unit').hexdigest(),source='dev/unit',sha256='a'*64,split='validation',label=0)
        (dev/'manifest.json').write_text(json.dumps(dict(variants=1,records=[row])))
        metadata=dict(candidate='hp160',frontend_version='adaptive',threshold_q10=0,accepted_validation=True,
                      weights_sha256=hashlib.sha256((model/'weights.npz').read_bytes()).hexdigest(),
                      dataset_manifest_sha256=hashlib.sha256((dev/'manifest.json').read_bytes()).hexdigest())
        (model/'validation.json').write_text(json.dumps(metadata))
        y=np.tile(np.arange(7),8);x=np.zeros((56,260),dtype=np.float32);x[np.arange(56),y]=4
        arrays=dict(validation=dict(x=x,y=y,profiles=[p for p in PROFILES for _ in range(7)]))
        return model,dev,arrays

    def test_freeze_recomputes_validation_and_does_not_trust_success_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model,dev,arrays=self.fixture(root)
            with patch.object(freeze,'make_arrays',return_value=arrays):
                result=freeze.freeze_channel(model,dev,root/'frozen')
            self.assertEqual(result['threshold_q10'],0)
            self.assertEqual(result['conditioning'],'hp160');self.assertEqual(result['frontend'],'adaptive')
            self.assertTrue(result['source_sha256']);self.assertFalse(result['final_evaluated'])
            self.assertEqual((model/'weights.npz').read_bytes(),(root/'frozen/weights.npz').read_bytes())
            arrays['validation']['x'][:]=0
            with patch.object(freeze,'make_arrays',return_value=arrays):
                with self.assertRaises(ValueError):freeze.freeze_channel(model,dev,root/'rejected')
            self.assertFalse((root/'rejected').exists())
            with self.assertRaises(ValueError):freeze.freeze_channel(model,dev,root/'frozen')

    def test_changed_sources_or_weights_and_overlapping_final_data_fail_before_pcm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model,dev,arrays=self.fixture(root)
            with patch.object(freeze,'make_arrays',return_value=arrays):freeze.freeze_channel(model,dev,root/'frozen')
            final=root/'final';final.mkdir();(final/'manifest.json').write_text((dev/'manifest.json').read_text())
            with patch.object(freeze,'_examples',side_effect=AssertionError('must refuse before PCM')):
                with self.assertRaises(ValueError):freeze.evaluate_channel_frozen(root/'frozen',final,root/'no-report')
            p=root/'frozen/channel-frozen.json';metadata=json.loads(p.read_text())
            metadata['source_sha256']['arcade_ai_pet/training.py']='0'*64;p.write_text(json.dumps(metadata))
            with patch.object(freeze,'_examples',side_effect=AssertionError('must refuse before PCM')):
                with self.assertRaises(ValueError):freeze.evaluate_channel_frozen(root/'frozen',final,root/'no-report')
            self.assertFalse((root/'no-report').exists())

    def test_final_evaluation_keeps_fixed_threshold_and_never_claims_device_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model,dev,arrays=self.fixture(root)
            with patch.object(freeze,'make_arrays',return_value=arrays):freeze.freeze_channel(model,dev,root/'frozen')
            final=root/'final';final.mkdir();records=[];examples=[]
            for label in range(7):
                source=f'final/{label}'
                records.append(dict(source=source,id=hashlib.sha256(source.encode()).hexdigest(),sha256=hashlib.sha256(('wave/'+source).encode()).hexdigest(),split='test',label=label))
                pcm=np.full(32000,label+1,dtype=np.int16);examples.append((pcm,label,None))
            (final/'manifest.json').write_text(json.dumps(dict(variants=1,records=records)))
            def extract(pcm,*args):
                x=np.zeros(260,dtype=np.float32);x[int(pcm[0])-1]=4;return x
            with patch.object(freeze,'_examples',return_value=iter(examples)),patch.object(freeze,'_channel_features',side_effect=extract),patch('arcade_ai_pet.channel_training.select_channel_threshold',side_effect=AssertionError('no retuning on final')):
                report=freeze.evaluate_channel_frozen(root/'frozen',final,root/'report')
            self.assertTrue(report['accepted_synthetic']);self.assertEqual(report['threshold_q10'],0)
            self.assertEqual(len(report['profiles']),8)
            self.assertTrue(all(r['examples']==7 for r in report['profiles']))
            self.assertFalse(report['real_voice_tested']);self.assertFalse(report['device_tested'])

    def test_equal_wave_under_new_id_and_changed_frozen_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model,dev,arrays=self.fixture(root)
            with patch.object(freeze,'make_arrays',return_value=arrays):freeze.freeze_channel(model,dev,root/'frozen')
            final=root/'final';final.mkdir()
            (final/'manifest.json').write_text(json.dumps(dict(variants=1,records=[dict(id='c'*64,source='different',sha256='a'*64)])))
            with patch.object(freeze,'_examples',side_effect=AssertionError('no PCM for relabelled duplicate')):
                with self.assertRaises(ValueError):freeze.evaluate_channel_frozen(root/'frozen',final,root/'no-report')
            with (root/'frozen/weights.npz').open('ab') as stream:stream.write(b'changed')
            with patch.object(freeze,'_examples',side_effect=AssertionError('no PCM for changed weights')):
                with self.assertRaises(ValueError):freeze.evaluate_channel_frozen(root/'frozen',final,root/'no-report')
            self.assertFalse((root/'no-report').exists())
