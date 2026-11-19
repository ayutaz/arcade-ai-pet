"""既存2113値を保持し、平均除去前の32bandを別の計算へ照合する。"""
import unittest
import numpy as np
from arcade_ai_pet import envelope_frontend as frontend
from arcade_ai_pet.logmel_frontend import features as previous
from arcade_ai_pet.features import WINDOW,HOP,FFT,FRAMES,_WINDOW,_MEL
from arcade_ai_pet.channel_frontend import active_span
from test_logmel_frontend import scalar_pool

class EnvelopeFrontendTests(unittest.TestCase):
    def test_exact_old_prefix_and_absolute_bands_match_independent_spectral_reference(self):
        pcm=np.zeros(32000,dtype=np.int16);pcm[5000:17000]=(2400*np.sin(np.arange(12000)*.19)).astype(np.int16);before=pcm.copy()
        actual=frontend.features(pcm)
        self.assertEqual(actual.dtype,np.float32);self.assertEqual(actual.shape,(2145,))
        np.testing.assert_array_equal(actual[:2113],previous(pcm))
        x=pcm.astype(np.float64)/float(abs(pcm.astype(np.float64)).max())
        frames=np.lib.stride_tricks.sliding_window_view(x,WINDOW)[::HOP]
        first,last=active_span(np.mean(frames**2,axis=1));first,last=max(0,first-10),min(FRAMES,last+10)
        logmel=np.log(np.maximum(abs(np.fft.rfft(frames*_WINDOW,n=FFT))**2@_MEL.T,1e-8))
        expected=np.clip(scalar_pool(logmel[first:last]).mean(0)/4,-16,16).astype(np.float32)
        np.testing.assert_allclose(actual[2113:],expected,rtol=0,atol=1e-6)
        self.assertGreater(float(abs(actual[2113:]).max()),.1)
        np.testing.assert_array_equal(pcm,before)

    def test_quiet_and_silent_inputs_keep_the_existing_zero_contract(self):
        for magnitude in (0,1,31):
            pcm=np.full(32000,magnitude,dtype=np.int16)
            np.testing.assert_array_equal(frontend.features(pcm),np.zeros(2145,dtype=np.float32))

    def test_boundaries_and_negative_full_scale_are_finite_without_changing_prefix(self):
        for position in (0,15999,31999):
            pcm=np.zeros(32000,dtype=np.int16);pcm[position]=-32768;before=pcm.copy()
            actual=frontend.features(pcm)
            self.assertTrue(np.isfinite(actual).all());self.assertLessEqual(float(abs(actual).max()),16)
            np.testing.assert_array_equal(actual[:2113],previous(pcm));np.testing.assert_array_equal(pcm,before)

    def test_invalid_pcm_refuses(self):
        for pcm in (np.zeros(31999,dtype=np.int16),np.zeros(32000),np.zeros((1,32000),dtype=np.int16),[0]*32000):
            with self.assertRaises(ValueError):frontend.features(pcm)

if __name__=='__main__':unittest.main()
