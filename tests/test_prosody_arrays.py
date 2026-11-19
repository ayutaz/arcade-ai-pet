"""原音を切らず、trainだけへ速度/音程の同時変化を加える。"""
import hashlib,json,tempfile,unittest,wave
from pathlib import Path
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import prosody_arrays as prosody


class ProsodyArraysTests(unittest.TestCase):
    def test_rate_changes_duration_preserves_endpoints_and_refuses_truncation(self):
        pcm=np.linspace(-2000,2000,8000).astype(np.int16)
        for rate in (.9,1.1):
            result=prosody.speed_variant(pcm,rate)
            self.assertEqual(len(result),round(len(pcm)/rate));self.assertEqual(result.dtype,np.int16)
            self.assertEqual(int(result[0]),int(pcm[0]));self.assertEqual(int(result[-1]),int(pcm[-1]))
        with self.assertRaises(ValueError):prosody.speed_variant(np.ones(32000,dtype=np.int16),.9)
        for invalid in (True,2.,float('nan')):
            with self.assertRaises(ValueError):prosody.speed_variant(pcm,invalid)

    def fixture(self,root,rate=16000):
        (root/'sources').mkdir();records=[]
        pcm=np.rint(2000*np.sin(np.arange(8000)*.1)).astype('<i2')
        for label,split in enumerate(('train','validation','test')):
            ident=hashlib.sha256(split.encode()).hexdigest();file=root/'sources'/(ident+'.wav')
            if split=='train':
                with wave.open(str(file),'wb') as stream:stream.setparams((1,2,rate,0,'NONE','not compressed'));stream.writeframes(pcm.tobytes())
            records.append(dict(id=ident,source=split,split=split,label=label,prompt=('あそぼ','おいで','まって')[label],sha256=hashlib.sha256(file.read_bytes()).hexdigest() if file.exists() else 'a'*64))
        manifest=dict(role='development',variants=1,records=records);(root/'manifest.json').write_text(json.dumps(manifest))
        return manifest,dict(train=dict(x=np.zeros((14,260),dtype=np.float32),y=np.zeros(14,dtype=np.int64),profiles=['clean']*14),validation=dict(x=np.ones((1,260),dtype=np.float32),y=np.ones(1,dtype=np.int64),profiles=['clean']))

    def test_neural_48khz_source_is_resampled_before_rate_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();manifest,base=self.fixture(root,rate=48000)
            qa=root/'qualified';plan=prosody.prepare_sources(root,qa)
            self.assertEqual(len(plan['transforms']),2)
            self.assertTrue(all(row['frames']<3200 for row in plan['transforms']))
            for row in plan['transforms']:
                with wave.open(str(qa/'sources'/row['file'])) as stream:self.assertEqual(stream.getframerate(),16000)
            report=prosody.qualify_sources(qa,recognize=lambda *a:'あそぼ')
            self.assertEqual(report['accepted_transforms'],2)

    def test_validation_is_unchanged_and_extra_rates_skip_validation_test_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();manifest,base=self.fixture(root);before=base['validation']['x'].copy()
            qa=root/'qualified';prosody.prepare_sources(root,qa);prosody.qualify_sources(qa,recognize=lambda *a:'あそぼ')
            with patch.object(prosody,'adaptive_arrays',return_value=base):
                result=prosody.make_arrays(root,'hp160',frontend='adaptive',noise_replicas=5,noise_mode='fresh',hum_replicas=3,qualified=qa)
            self.assertEqual(len(result['train']['y']),42);self.assertEqual(set(result['train']['y']),{0})
            np.testing.assert_array_equal(result['validation']['x'],before)
            self.assertEqual(len(base['train']['y']),14);self.assertTrue(np.isfinite(result['train']['x']).all())
            self.assertEqual(result['augmentation']['independent_sources_added'],0)
            self.assertEqual(result['augmentation']['added_training_rows'],28)

    def test_final_role_and_changed_wave_fail_instead_of_silently_skipping(self):
        for mode in ('final','changed'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();manifest,base=self.fixture(root)
                qa=root/'qualified';prosody.prepare_sources(root,qa);prosody.qualify_sources(qa,recognize=lambda *a:'あそぼ')
                if mode=='final':manifest['role']='final';(root/'manifest.json').write_text(json.dumps(manifest))
                else:(root/'sources'/(manifest['records'][0]['id']+'.wav')).write_bytes(b'changed')
                with patch.object(prosody,'adaptive_arrays',return_value=base):
                    with self.assertRaises(ValueError):prosody.make_arrays(root,'hp160',qualified=qa)

    def test_unqualified_rate_variants_are_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();manifest,base=self.fixture(root)
            with patch.object(prosody,'adaptive_arrays',return_value=base):
                with self.assertRaises(ValueError):prosody.make_arrays(root,'hp160')

    def test_content_mismatch_is_excluded_and_qualification_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();manifest,base=self.fixture(root);qa=root/'qualified';prosody.prepare_sources(root,qa)
            replies=iter(('あそぼ','まって'));report=prosody.qualify_sources(qa,recognize=lambda *a:next(replies))
            self.assertEqual(report['accepted_transforms'],1);self.assertEqual(report['rejected_transforms'],1)
            with patch.object(prosody,'adaptive_arrays',return_value=base):
                result=prosody.make_arrays(root,'hp160',noise_replicas=5,hum_replicas=3,qualified=qa)
            self.assertEqual(len(result['train']['y']),28);self.assertEqual(result['augmentation']['accepted_transforms'],1)
            with self.assertRaises(ValueError):prosody.prepare_sources(root,qa)
            with self.assertRaises(ValueError):prosody.qualify_sources(qa,recognize=lambda *a:'あそぼ')
            plan=json.loads((qa/'plan.json').read_text());(qa/'sources'/plan['transforms'][0]['file']).write_bytes(b'changed')
            with patch.object(prosody,'adaptive_arrays',return_value=base):
                with self.assertRaises(ValueError):prosody.make_arrays(root,'hp160',qualified=qa)

    def test_recognizer_error_cannot_produce_training_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();manifest,base=self.fixture(root);qa=root/'qualified';prosody.prepare_sources(root,qa)
            def fail(*args):raise RuntimeError('private provider detail')
            report=prosody.qualify_sources(qa,recognize=fail)
            self.assertFalse(report['complete']);self.assertNotIn('private',json.dumps(report))
            with patch.object(prosody,'adaptive_arrays',return_value=base):
                with self.assertRaises(ValueError):prosody.make_arrays(root,'hp160',qualified=qa)


if __name__=='__main__':unittest.main()
