import unittest
import numpy as np
from arcade_ai_pet import channel_robustness as channel


class ChannelRobustnessTests(unittest.TestCase):
    def test_profiles_are_deterministic_bounded_and_do_not_modify_source(self):
        source=np.zeros(32000,dtype=np.int16)
        source[4000:12000]=(3000*np.sin(np.arange(8000)*.13)).astype(np.int16)
        original=source.copy()
        for profile in channel.PROFILES:
            with self.subTest(profile=profile):
                first=channel.simulate(source,profile,41)
                self.assertEqual(first.shape,(32000,));self.assertEqual(first.dtype,np.int16)
                np.testing.assert_array_equal(first,channel.simulate(source,profile,41))
        np.testing.assert_array_equal(source,original)
        np.testing.assert_array_equal(channel.simulate(source,'clean',41),source)
        with self.assertRaises(ValueError):channel.simulate(source,'device-measured',41)

    def test_centering_matches_c_truncation_clamping_and_extremes(self):
        pcm=np.full(32000,1600,dtype=np.int16)
        np.testing.assert_array_equal(channel.condition(pcm,'center'),np.zeros(32000,dtype=np.int16))
        pcm[0]=-32768
        expected=np.clip(pcm.astype(np.int64)-int(pcm.astype(np.int64).sum()/32000),-32768,32767).astype(np.int16)
        np.testing.assert_array_equal(channel.condition(pcm,'center'),expected)
        np.testing.assert_array_equal(channel.condition(pcm,'raw'),pcm)
        for invalid in (np.zeros(32000),np.zeros(31999,dtype=np.int16)):
            with self.assertRaises(ValueError):channel.condition(invalid,'center')

    def test_highpass_rejects_dc_preserves_speech_band_and_has_no_cross_record_state(self):
        t=np.arange(32000)/16000
        for name in ('hp20','hp80','hp160'):
            filtered=channel.condition(np.full(32000,1700,dtype=np.int16),name)
            np.testing.assert_array_equal(filtered,np.zeros(32000,dtype=np.int16))
        source=(1600+3000*np.sin(2*np.pi*1000*t)+500*np.sin(2*np.pi*2*t)).astype(np.int16)
        a=channel.condition(source,'hp80');b=channel.condition(source,'hp80')
        np.testing.assert_array_equal(a,b)
        ac=a[1600:].astype(float)
        def amplitude(hz):return abs(np.mean(ac*np.exp(-2j*np.pi*hz*t[1600:]))) *2
        self.assertGreater(amplitude(1000),2700);self.assertLess(amplitude(2),25)
        with self.assertRaises(ValueError):channel.condition(source,'hp-device')

    def test_validation_selection_requires_all_profiles_and_quality_gates(self):
        rows=[]
        for candidate in channel.CANDIDATES:
            for profile in channel.PROFILES:
                rows.append(dict(candidate=candidate,profile=profile,macro_f1=.95,
                                 false_activation_rate=.01,command_recall=.95))
        self.assertEqual(channel.choose(rows),'center')
        for row in rows:
            if row['candidate']=='center' and row['profile']=='drift':row['macro_f1']=.1
        self.assertEqual(channel.choose(rows),'hp20')
        for row in rows:row['false_activation_rate']=.5
        self.assertIsNone(channel.choose(rows))
        with self.assertRaises(ValueError):channel.choose(rows[:-1])
        with self.assertRaises(ValueError):channel.choose(rows+[rows[0]])

    def test_source_ids_cannot_cross_development_and_final_corpora(self):
        self.assertTrue(channel.check_disjoint({'voice/a','noise/1'},{'voice/b','noise/2'}))
        with self.assertRaises(ValueError):channel.check_disjoint({'voice/a'},{'voice/a'})
        with self.assertRaises(ValueError):channel.check_disjoint(set(),{'voice/a'})

    def test_metric_keeps_negative_false_activation_separate_from_command_recall(self):
        result=channel.measure([0,1,5,6],[0,5,1,6])
        self.assertEqual(result['command_recall'],.5)
        self.assertEqual(result['false_activation_rate'],.5)
        self.assertEqual(result['examples'],4)
        with self.assertRaises(ValueError):channel.measure([0],[7])
        with self.assertRaises(ValueError):channel.measure([],[])

    def test_comparison_counts_every_profile_and_does_not_claim_device_evidence(self):
        examples=[(np.zeros(32000,dtype=np.int16),6),(np.full(32000,900,dtype=np.int16),5)]
        rows=channel.compare(examples,lambda pcm:6)
        self.assertEqual(len(rows),len(channel.CANDIDATES)*len(channel.PROFILES))
        for row in rows:
            self.assertEqual(row['examples'],2)
            self.assertEqual(row['false_activation_rate'],0)
        with self.assertRaises(ValueError):channel.compare(examples,lambda pcm:-1)

    def test_runner_refuses_overlap_and_defers_final_waveforms_when_no_candidate_passes(self):
        from pathlib import Path
        import hashlib,json,tempfile
        from unittest.mock import patch
        import contextlib
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,source in [('dev','source/a'),('final','source/b')]:
                path=root/name;path.mkdir()
                record=dict(id=hashlib.sha256(source.encode()).hexdigest(),source=source,label=0,split='validation',sha256=hashlib.sha256(source.encode()).hexdigest())
                (path/'manifest.json').write_text(json.dumps(dict(variants=1,records=[record])))
            calls=[]
            def examples(path):
                calls.append(Path(path).name)
                yield np.zeros(32000,dtype=np.int16),0,np.zeros(260,dtype=np.float32)
            class Model:
                weights_sha256='a'*64
                def extract(self,pcm):return np.zeros(260,dtype=np.float32)
                def predict(self,features):return 6,np.zeros(7,dtype=np.int32)
            with patch.object(channel,'_examples',side_effect=examples),patch.object(channel,'NativeModel',return_value=contextlib.nullcontext(Model())):
                report=channel.evaluate_channels(root/'assets',root/'dev',root/'final',root/'report')
            self.assertEqual(calls,['dev'])
            self.assertIsNone(report['selected_candidate']);self.assertFalse(report['final_evaluated'])
            self.assertFalse(report['device_tested']);self.assertFalse(report['real_voice_tested'])
            self.assertTrue((root/'report/channel-evaluation.json').is_file())
            with self.assertRaises(ValueError):channel.evaluate_channels(root/'assets',root/'dev',root/'dev',root/'other')
            self.assertFalse((root/'other').exists())

    def test_native_conditioner_rejects_invalid_buffers_before_writing_and_clamps_extremes(self):
        import ctypes as C
        fn=channel._filter()[1]
        pcm=np.tile(np.array([-32768,32767],dtype=np.int16),16000)
        before=pcm.copy()
        self.assertEqual(fn(None,32000,80),-1)
        self.assertEqual(fn(pcm.ctypes.data,31999,80),-1)
        self.assertEqual(fn(pcm.ctypes.data,32000,10),-1)
        np.testing.assert_array_equal(pcm,before)
        for name in channel.CANDIDATES:
            out=channel.condition(pcm,name)
            self.assertEqual(out.dtype,np.int16)
            self.assertGreater(out.astype(np.int64).var(),1000)

    def test_identical_wave_hash_is_rejected_even_with_different_source_ids(self):
        from pathlib import Path
        import json,tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('a','b'):
                path=root/name;path.mkdir()
                (path/'manifest.json').write_text(json.dumps(dict(variants=1,records=[dict(source=name,sha256='a'*64)])))
            with self.assertRaisesRegex(ValueError,'waveform overlap'):
                channel.evaluate_channels(root/'assets',root/'a',root/'b',root/'report')
            self.assertFalse((root/'report').exists())
