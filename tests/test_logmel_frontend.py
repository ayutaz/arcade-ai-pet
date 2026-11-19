"""64×32 log-melとfractional poolingを独立した区間積分と照合する。"""
import unittest
import numpy as np
from arcade_ai_pet import logmel_frontend as frontend
from arcade_ai_pet.features import WINDOW,HOP,FFT,FRAMES,_WINDOW,_MEL
from arcade_ai_pet.channel_frontend import active_span
from arcade_ai_pet.duration_frontend import features as previous


def scalar_pool(values):
    width=len(values)/64
    rows=[]
    for b in range(64):
        left,right=b*width,(b+1)*width
        rows.append(sum(values[i]*max(0.,min(right,i+1)-max(left,i)) for i in range(len(values)))/width)
    return np.asarray(rows)


class LogmelFrontendTests(unittest.TestCase):
    def test_fractional_pool_has_no_empty_bins_and_conserves_mean(self):
        rng=np.random.default_rng(27)
        for frames in (1,2,3,20,40,64,198):
            values=rng.normal(size=(frames,3));before=values.copy()
            actual=frontend.pool(values)
            self.assertEqual(actual.shape,(64,3))
            np.testing.assert_allclose(actual,scalar_pool(values),rtol=0,atol=2e-13)
            np.testing.assert_allclose(actual.mean(0),values.mean(0),rtol=0,atol=1e-13)
            np.testing.assert_array_equal(values,before)
            self.assertFalse(np.shares_memory(actual,values))

    def test_fractional_pool_preserves_constant_bands_and_rejects_bad_inputs(self):
        values=np.tile(np.array([2.,5.,-3.]),(37,1))
        np.testing.assert_allclose(frontend.pool(values),np.tile(values[0],(64,1)),rtol=0,atol=1e-14)
        for bad in (np.zeros((0,3)),np.zeros(3),np.zeros((3,0)),np.array([[np.nan]]),np.array([[np.inf]]),[[1.]]):
            with self.assertRaises(ValueError):frontend.pool(bad)

    def test_full_frontend_matches_spectrum_reference_and_keeps_duration_semantics(self):
        pcm=np.zeros(32000,dtype=np.int16)
        pcm[6000:18000]=(1600*np.sin(np.arange(12000)*.31)).astype(np.int16)
        before=pcm.copy();actual=frontend.features(pcm)
        x=pcm.astype(np.float64)/abs(pcm.astype(np.float64)).max()
        frames=np.lib.stride_tricks.sliding_window_view(x,WINDOW)[::HOP]
        energy=np.mean(frames**2,axis=1);first,last=active_span(energy);duration=4*(last-first)/FRAMES
        first,last=max(0,first-10),min(FRAMES,last+10)
        power=abs(np.fft.rfft(frames*_WINDOW,n=FFT))**2
        bands=scalar_pool(np.log(np.maximum(power@_MEL.T,1e-8))[first:last])
        bands=(bands-bands.mean(0))/4
        log_energy=np.log(np.maximum(energy,1e-8));log_energy-=log_energy.max()
        energies=scalar_pool(log_energy[first:last,None]).ravel()/2
        expected=np.clip(np.concatenate((bands.ravel(),energies,[duration])),-16,16).astype(np.float32)
        self.assertEqual(actual.dtype,np.float32);self.assertEqual(actual.shape,(2113,))
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-6)
        self.assertEqual(float(actual[-1]),float(previous(pcm,context=True)[-1]))
        np.testing.assert_allclose(actual[:2048].reshape(64,32).mean(0),0,rtol=0,atol=1e-6)
        np.testing.assert_array_equal(pcm,before)

    def test_silence_is_all_zero_and_short_or_boundary_impulses_are_finite(self):
        np.testing.assert_array_equal(frontend.features(np.zeros(32000,dtype=np.int16)),np.zeros(2113,dtype=np.float32))
        for pos in (0,15999,31919,31999):
            pcm=np.zeros(32000,dtype=np.int16);pcm[pos]=3000
            values=frontend.features(pcm)
            self.assertTrue(np.isfinite(values).all());self.assertLessEqual(float(abs(values).max()),16)
            self.assertGreaterEqual(float(values[-1]),0);self.assertLessEqual(float(values[-1]),4)
            self.assertEqual(float(values[-1]),float(previous(pcm,context=True)[-1]))

    def test_full_scale_negative_pcm_never_overflows_or_changes_the_input(self):
        pcm=np.tile(np.array([-32768,32767],dtype=np.int16),16000);before=pcm.copy()
        actual=frontend.features(pcm)
        self.assertTrue(np.isfinite(actual).all());self.assertLessEqual(float(abs(actual).max()),16)
        np.testing.assert_array_equal(pcm,before)

    def test_wrong_pcm_contract_is_rejected(self):
        for bad in (np.zeros(31999,dtype=np.int16),np.zeros(32000,dtype=np.int32),np.zeros((1,32000),dtype=np.int16),[0]*32000):
            with self.assertRaises(ValueError):frontend.features(bad)


if __name__=='__main__':unittest.main()
