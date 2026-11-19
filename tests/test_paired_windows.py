"""先行契約: 同一の開発音声を対にし、VAD境界と特徴差だけを調べる。"""
import json
import unittest
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import paired_windows as diagnostic
from arcade_ai_pet.logmel_frontend import features
from arcade_ai_pet.features import _WINDOW,_MEL
from arcade_ai_pet.channel_robustness import simulate


def fixture(start=8000,length=6000):
    pcm=np.zeros(32000,dtype=np.int16)
    pcm[start:start+length]=np.rint(7000*np.sin(np.arange(length)*.073)).astype(np.int16)
    return pcm


def scalar_span(pcm):
    peak=max(abs(int(v)) for v in pcm)
    if peak<32:return None
    energy=[sum((int(v)/peak)**2 for v in pcm[i*160:i*160+400])/400 for i in range(198)]
    limit=max(.0004,max(energy)*.06,min(sorted(energy)[39]*3,max(energy)*.8))
    active=[i for i,e in enumerate(energy) if e>=limit]
    first,last=(active[0],active[-1]+1) if active else (0,198)
    if last-first<20:first=max(0,min(first,178));last=first+20
    return first,last


class PairedWindowTests(unittest.TestCase):
    def test_trace_matches_independent_energy_window_and_clipped_context(self):
        for pcm in (fixture(),fixture(0),fixture(26000),np.full(32000,-32768,dtype=np.int16)):
            before=pcm.copy();row=diagnostic.trace(pcm);first,last=scalar_span(pcm)
            left,right=max(0,first-10),min(198,last+10)
            self.assertEqual(row['active_span'],[first,last]);self.assertEqual(row['context_span'],[left,right])
            self.assertEqual(row['context_clipped_frames'],[max(0,10-first),max(0,last+10-198)])
            self.assertEqual(row['selected_samples'],[left*160,(right-1)*160+400])
            self.assertEqual(row['peak_amplitude'],max(abs(int(v)) for v in pcm));self.assertFalse(row['features_skipped'])
            np.testing.assert_array_equal(pcm,before)

    def test_explicit_span_matches_existing_features_and_independent_area_pool(self):
        for pcm in (fixture(),fixture(0),fixture(26000),np.full(32000,-32768,dtype=np.int16)):
            first,last=scalar_span(pcm)
            np.testing.assert_array_equal(diagnostic.for_span(pcm,first=first,last=last),features(pcm))
        pcm=fixture();first,last=42,75;x=pcm.astype(np.float64)/7000
        framed=np.stack([x[i*160:i*160+400] for i in range(198)])
        energy=np.mean(framed**2,axis=1);mel=np.log(np.maximum(abs(np.fft.rfft(framed*_WINDOW,n=512))**2@_MEL.T,1e-8))
        left,right=first-10,last+10;n=right-left
        def pool(values):
            out=np.zeros((64,values.shape[1]))
            for target in range(64):
                for frame in range(n):
                    overlap=max(0,min((target+1)*n,(frame+1)*64)-max(target*n,frame*64))
                    out[target]+=values[left+frame]*overlap/n
            return out
        bands=pool(mel);bands=(bands-bands.mean(0))/4
        log_energy=np.log(np.maximum(energy,1e-8));log_energy-=log_energy.max()
        expected=np.clip(np.concatenate((bands.ravel(),pool(log_energy[:,None]).ravel()/2,[4*(last-first)/198])),-16,16).astype(np.float32)
        np.testing.assert_allclose(diagnostic.for_span(pcm,first=first,last=last),expected,atol=2e-6,rtol=0)

    def test_shift_alignment_distinguishes_position_change_from_frame_loss(self):
        reference=fixture()
        for profile,shift in (('early',-15),('late',15)):
            changed=simulate(reference,profile,12);row=diagnostic.pair(reference,changed,shift_frames=shift)
            first,last=scalar_span(reference)
            self.assertEqual(row['aligned_reference_span'],[first+shift,last+shift])
            self.assertEqual(row['aligned_reference_clipped_frames'],0);self.assertEqual(row['active_boundary_l1_frames'],0)
            self.assertEqual(row['counterfactual_to_reference_rmse'],0)
        edge=fixture(0);row=diagnostic.pair(edge,simulate(edge,'early',12),shift_frames=-15)
        self.assertEqual(row['aligned_reference_span'][0],0);self.assertEqual(row['aligned_reference_clipped_frames'],15)

    def test_pair_reports_actual_feature_deltas_without_mutation_or_pcm_output(self):
        reference=fixture();changed=reference.copy();changed[:5000]=1700
        before=(reference.copy(),changed.copy());row=diagnostic.pair(reference,changed)
        a,b=features(reference).astype(np.float64),features(changed).astype(np.float64)
        self.assertAlmostEqual(row['logmel_rmse'],float(np.sqrt(np.mean((a[:2048]-b[:2048])**2))))
        self.assertAlmostEqual(row['energy_rmse'],float(np.sqrt(np.mean((a[2048:2112]-b[2048:2112])**2))))
        self.assertEqual(row['duration_delta'],float(b[-1]-a[-1]));self.assertGreater(row['active_boundary_l1_frames'],0)
        def metadata_only(value):
            if isinstance(value,dict):
                self.assertNotIn('pcm',value)
                for child in value.values():metadata_only(child)
            elif isinstance(value,list):
                self.assertLessEqual(len(value),2)
                for child in value:metadata_only(child)
            else:self.assertIsInstance(value,(str,int,float,bool,type(None)))
        metadata_only(row);json.dumps(row,allow_nan=False)
        np.testing.assert_array_equal(reference,before[0]);np.testing.assert_array_equal(changed,before[1])

    def test_low_peak_is_explicitly_skipped_without_fictitious_reference_window(self):
        for pcm in (np.zeros(32000,dtype=np.int16),np.full(32000,31,dtype=np.int16)):
            row=diagnostic.trace(pcm);self.assertTrue(row['features_skipped']);self.assertIsNone(row['active_span'])
            pair=diagnostic.pair(pcm,fixture());self.assertIsNone(pair['aligned_reference_span'])
            self.assertIsNone(pair['counterfactual_to_reference_rmse']);self.assertIsNone(pair['active_boundary_l1_frames'])
            np.testing.assert_array_equal(diagnostic.for_span(pcm,first=0,last=20),np.zeros(2113,dtype=np.float32))

    def test_invalid_pcm_span_or_shift_fails_before_fft(self):
        pcm=fixture()
        with patch('numpy.fft.rfft',side_effect=AssertionError('FFT reached')):
            for bad in ([1],pcm.astype(np.int32),pcm[:-1],np.zeros((16000,2),dtype=np.int16)):
                with self.assertRaises(ValueError):diagnostic.trace(bad)
                with self.assertRaises(ValueError):diagnostic.pair(pcm,bad)
            for first,last in ((True,20),(-1,20),(20,20),(20,199),(0,20.0)):
                with self.assertRaises(ValueError):diagnostic.for_span(pcm,first=first,last=last)
            for shift in (True,1,-16,15.0):
                with self.assertRaises(ValueError):diagnostic.pair(pcm,pcm,shift_frames=shift)

if __name__=='__main__':unittest.main()
