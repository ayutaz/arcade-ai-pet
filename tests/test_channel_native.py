import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import channel_native as native
from arcade_ai_pet.channel_freeze import freeze_channel
from arcade_ai_pet.channel_training import level
from arcade_ai_pet.channel_robustness import condition
from arcade_ai_pet.channel_frontend import features
from arcade_ai_pet.training import scores


class ChannelNativeTests(unittest.TestCase):
    def fixture(self,root):
        from test_channel_freeze import ChannelFreezeTests
        model,dev,arrays=ChannelFreezeTests().fixture(root)
        with patch('arcade_ai_pet.channel_freeze.make_arrays',return_value=arrays):freeze_channel(model,dev,root/'frozen')
        return native.export_channel(root/'frozen',root/'assets')

    def test_export_is_channel_specific_and_forged_header_cannot_be_compiled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);metadata=self.fixture(root)
            self.assertEqual(metadata['runtime'],'channel-adaptive-v1')
            self.assertFalse((root/'assets/metadata.json').exists())
            self.assertEqual(metadata['input_contract']['conditioning'],'hp160')
            self.assertEqual(metadata['input_contract']['peak'],4096)
            with self.assertRaises(ValueError):native.export_channel(root/'frozen',root/'assets')
            (root/'assets/pet_model.h').write_text('#include "/tmp/not-trusted"')
            metadata['header_sha256']['pet_model.h']=hashlib.sha256((root/'assets/pet_model.h').read_bytes()).hexdigest()
            (root/'assets/channel-metadata.json').write_text(json.dumps(metadata))
            with patch('arcade_ai_pet.channel_native.subprocess.run',side_effect=AssertionError('must reject before compiler')):
                with self.assertRaises(ValueError):native.ChannelNativeModel(root/'assets')

    def test_banked_and_contiguous_preparation_matches_frozen_pcm_bit_for_bit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            t=np.arange(32000)/16000
            fixtures=[np.zeros(32000,dtype=np.int16),np.full(32000,1600,dtype=np.int16),
                      np.tile(np.array([-32768,32767],dtype=np.int16),16000),
                      np.clip(1600+480*np.exp(-t/.3)+45*np.sin(2*np.pi*600*t),-32768,32767).astype(np.int16)]
            with native.ChannelNativeModel(root/'assets') as model:
                for pcm in fixtures:
                    before=pcm.copy();expected=level(condition(pcm,'hp160'))
                    for banked in (False,True):
                        np.testing.assert_array_equal(model.prepare(pcm,banked=banked),expected)
                    np.testing.assert_array_equal(pcm,before)
                self.assertLessEqual(model.scratch_bytes,24576)

    def test_adaptive_features_and_integer_engine_match_python_without_mutating_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);t=np.arange(32000)/16000
            rng=np.random.default_rng(71)
            for offset in (3200,11200):
                pcm=rng.normal(0,20,32000);pcm[offset:offset+8000]+=300*np.sin(2*np.pi*720*t[:8000])
                pcm=np.rint(pcm+1600).astype(np.int16);before=pcm.copy()
                with native.ChannelNativeModel(root/'assets') as model:
                    expected=features(level(condition(pcm,'hp160')))
                    for banked in (False,True):
                        actual=model.extract(pcm,banked=banked)
                        self.assertLess(float(abs(actual-expected).max()),.001)
                    label,logits=model.predict(expected)
                    reference=scores(expected[None,:],model.layers)[0]
                    np.testing.assert_array_equal(logits,reference)
                    self.assertEqual(label,int(reference.argmax()))
                np.testing.assert_array_equal(pcm,before)

    def test_native_channel_has_no_allocator_or_library_sort_dependency(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            with native.ChannelNativeModel(root/'assets') as model:
                symbols=subprocess.check_output(['nm','-u',model.lib._name],text=True)
                self.assertNotRegex(symbols,r'\b_?(?:qsort|malloc|calloc|realloc|free)(?:@|\b)')

    def test_full_comparison_separates_native_conformance_from_model_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);meta=self.fixture(root);dataset=root/'final';dataset.mkdir()
            rows=[];examples=[]
            for label in range(7):
                source=f'channel-final/{label}'
                rows.append(dict(id=hashlib.sha256(source.encode()).hexdigest(),source=source,sha256=hashlib.sha256(('wave/'+source).encode()).hexdigest(),label=label,split='test'))
                pcm=(1600+500*np.sin(np.arange(32000)*.1)).astype(np.int16)
                examples.append((pcm,label,None))
            (dataset/'manifest.json').write_text(json.dumps(dict(variants=1,records=rows)))
            reference=dict(weights_sha256=meta['weights_sha256'],frozen_sha256=meta['frozen_sha256'],
                           dataset_manifest_sha256=hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest())
            (root/'reference.json').write_text(json.dumps(reference))
            with patch.object(native,'_examples',return_value=iter(examples)):
                result=native.evaluate_native_channel(root/'frozen',root/'assets',dataset,root/'reference.json',root/'report')
            self.assertEqual(result['examples_per_profile'],7);self.assertEqual(result['total_profile_examples'],56)
            self.assertTrue(result['conformance_passed']);self.assertFalse(result['quality_passed'])
            self.assertFalse(result['device_tested']);self.assertFalse(result['real_voice_tested'])
            reference['dataset_manifest_sha256']='0'*64;(root/'reference.json').write_text(json.dumps(reference))
            with patch.object(native,'_examples',side_effect=AssertionError('no mismatched PCM')):
                with self.assertRaises(ValueError):native.evaluate_native_channel(root/'frozen',root/'assets',dataset,root/'reference.json',root/'refused')
