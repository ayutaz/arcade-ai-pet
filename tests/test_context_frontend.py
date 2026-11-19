"""弱い語頭が特徴量から消えないためのホストPoC。"""
import unittest
import numpy as np
from arcade_ai_pet import context_frontend as context
from arcade_ai_pet import channel_frontend as legacy


class ContextFrontendTests(unittest.TestCase):
    def test_window_retains_ten_context_frames_and_stays_in_recording(self):
        energy=np.zeros(198);energy[50:70]=1
        self.assertEqual(legacy.active_span(energy),(50,70));self.assertEqual(context.active_span(energy),(40,80))
        energy[:]=1;self.assertEqual(context.active_span(energy),(0,198))
        with self.assertRaises(ValueError):context.active_span(np.full(198,np.nan))

    def test_weak_prefix_changes_context_features_when_legacy_discards_it(self):
        strong=np.zeros(32000,dtype=np.int16)
        strong[16000:19200]=np.rint(10000*np.sin(2*np.pi*500*np.arange(3200)/16000)).astype(np.int16)
        weak=strong.copy();weak[15200:15600]=np.rint(1000*np.sin(2*np.pi*2500*np.arange(400)/16000)).astype(np.int16)
        np.testing.assert_array_equal(legacy.features(strong),legacy.features(weak))
        self.assertGreater(float(abs(context.features(strong)-context.features(weak)).max()),.01)

    def test_zero_and_input_contract_keep_the_same_260_value_shape(self):
        x=context.features(np.zeros(32000,dtype=np.int16));self.assertEqual(x.shape,(260,));self.assertEqual(x.dtype,np.float32)
        self.assertTrue(np.array_equal(x,np.zeros(260)))
        for invalid in (np.zeros(32000,dtype=float),np.zeros(3200,dtype=np.int16)):
            with self.assertRaises(ValueError):context.features(invalid)


if __name__=='__main__':unittest.main()
