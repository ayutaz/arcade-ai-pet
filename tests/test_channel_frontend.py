import unittest
import numpy as np
from arcade_ai_pet import channel_frontend as frontend


class ChannelFrontendTests(unittest.TestCase):
    def test_adaptive_span_excludes_stationary_background_and_preserves_minimum_frames(self):
        energy=np.full(198,.01);energy[60:100]=.1
        self.assertEqual(frontend.active_span(energy),(60,100))
        self.assertEqual(frontend.active_span(np.ones(198)),(0,198))
        self.assertEqual(frontend.active_span(np.zeros(198)),(0,198))
        energy=np.zeros(198);energy[-1]=1
        self.assertEqual(frontend.active_span(energy),(178,198))
        for bad in (np.zeros(197),np.full(198,np.nan),np.full(198,-1)):
            with self.assertRaises(ValueError):frontend.active_span(bad)

    def test_stationary_subtraction_preserves_added_tone_and_has_a_positive_floor(self):
        power=np.full((198,257),4.0);power[60:100,10]=104
        energy=np.ones(198);energy[60:100]=10
        original=power.copy();result=frontend.subtract_noise(power,energy)
        self.assertAlmostEqual(result[70,10],100)
        self.assertAlmostEqual(result[0,10],.2)
        np.testing.assert_array_equal(power,original)
        self.assertTrue(np.isfinite(result).all());self.assertTrue((result>=0).all())
        with self.assertRaises(ValueError):frontend.subtract_noise(power[:1],energy)

    def test_features_are_bounded_silence_safe_and_do_not_mutate_pcm(self):
        for denoise in (False,True):
            zero=np.zeros(32000,dtype=np.int16)
            np.testing.assert_array_equal(frontend.features(zero,denoise=denoise),np.zeros(260,dtype=np.float32))
            x=(2000*np.sin(2*np.pi*600*np.arange(32000)/16000)).astype(np.int16)
            before=x.copy();values=frontend.features(x,denoise=denoise)
            self.assertEqual(values.shape,(260,));self.assertEqual(values.dtype,np.float32)
            self.assertTrue(np.isfinite(values).all());self.assertLessEqual(abs(values).max(),16)
            np.testing.assert_array_equal(x,before)
        with self.assertRaises(ValueError):frontend.features(np.zeros(32000))
