"""同じ音声処理の各段階と実特徴の切出し範囲を先に定義する。"""
import unittest
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import utterance_stages as stages
from arcade_ai_pet.data import trim_silence,augment
from arcade_ai_pet.channel_robustness import condition
from arcade_ai_pet.channel_training import level


def fixture():
    pcm=np.zeros(24000,dtype=np.int16)
    pcm[6000:12000]=np.rint(17000*np.sin(np.arange(6000)*.07)).astype(np.int16)
    return pcm


def reference_range(pcm):
    x=pcm.astype(np.float64);x/=max(abs(x))
    energy=np.array([sum(float(v)*float(v) for v in x[i*160:i*160+400])/400 for i in range(198)])
    maximum=max(energy);baseline=sorted(energy)[39]
    limit=max(.0004,maximum*.06,min(baseline*3,maximum*.8))
    active=[i for i,e in enumerate(energy) if e>=limit]
    first,last=(active[0],active[-1]+1) if active else (0,198)
    if last-first<20:first=max(0,min(first,178));last=first+20
    left,right=max(0,first-10),min(198,last+10)
    return first,last,left*160,(right-1)*160+400


class UtteranceStagesTests(unittest.TestCase):
    def test_fixed_eleven_stages_match_existing_augmentation_and_conditioning(self):
        pcm=fixture();before=pcm.copy();rows=stages.prepare(pcm,seed=711)
        self.assertEqual(len(rows),11)
        self.assertEqual([(r['stage'],r['variant']) for r in rows],
                         [('resampled',None),('trimmed',None)]+[(name,v) for v in range(3) for name in ('augmented','conditioned','cropped')])
        np.testing.assert_array_equal(rows[0]['pcm'],pcm)
        np.testing.assert_array_equal(rows[1]['pcm'],trim_silence(pcm))
        for variant in range(3):
            expected=augment(trim_silence(pcm),711+variant)
            np.testing.assert_array_equal(rows[2+3*variant]['pcm'],expected)
            np.testing.assert_array_equal(rows[3+3*variant]['pcm'],level(condition(expected,'hp160')))
        np.testing.assert_array_equal(pcm,before)
        rows[0]['pcm'][0]=100
        np.testing.assert_array_equal(pcm,before)

    def test_crop_matches_independent_energy_limits_and_frame_extent(self):
        for pcm in (fixture(),np.full(32000,-32768,dtype=np.int16)):
            rows=stages.prepare(pcm,seed=52)
            for variant in range(3):
                conditioned=rows[3+variant*3]['pcm'];cropped=rows[4+variant*3]
                first,last,start,end=reference_range(conditioned)
                self.assertEqual((cropped['active_first_frame'],cropped['active_last_frame'],cropped['start_sample'],cropped['end_sample']),(first,last,start,end))
                np.testing.assert_array_equal(cropped['pcm'],conditioned[start:end])
                self.assertTrue(0<=start<end<=32000)

    def test_fixed_audit_gain_handles_full_scale_silence_and_keeps_source_unchanged(self):
        pcm=np.array([-32768,-16384,0,16384,32767],dtype=np.int16);before=pcm.copy()
        np.testing.assert_array_equal(stages.normalize_for_asr(pcm),[-16384,-8192,0,8192,16384])
        np.testing.assert_array_equal(pcm,before)
        np.testing.assert_array_equal(stages.normalize_for_asr(np.zeros(2000,dtype=np.int16)),np.zeros(2000,dtype=np.int16))
        long=np.full(96000,1000,dtype=np.int16)
        self.assertTrue(np.all(stages.normalize_for_asr(long)==16384))
        with self.assertRaises(ValueError):stages.normalize_for_asr(long.astype(float))

    def test_invalid_source_and_seed_refuse_before_augmentation(self):
        for pcm,seed in ((fixture().astype(float),0),(np.zeros(32000,dtype=np.int16),0),
                         (np.ones(32001,dtype=np.int16),0),(np.ones(1000,dtype=np.int16),0),
                         (fixture(),-1),(fixture(),True),(fixture(),2**32)):
            with patch('arcade_ai_pet.utterance_stages.augment',side_effect=AssertionError('augmentation reached')):
                with self.assertRaises(ValueError):stages.prepare(pcm,seed=seed)


if __name__=='__main__':unittest.main()
