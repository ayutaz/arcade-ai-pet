import unittest
import numpy as np
from arcade_ai_pet.features import features, split_for


class FeatureTests(unittest.TestCase):
    def test_speech_alignment_removes_padding_dependence(self):
        x=np.zeros(32000,dtype=np.int16);y=x.copy()
        voice=(3000*np.sin(2*np.pi*650*np.arange(8000)/16000)).astype(np.int16)
        x[1600:9600]=voice;y[14400:22400]=voice
        self.assertLess(np.max(abs(features(x)-features(y))),.001)

    def test_fixed_bounded_features_and_silence(self):
        zero=features(np.zeros(32000,dtype=np.int16))
        self.assertEqual(zero.shape,(260,));self.assertTrue(np.isfinite(zero).all())
        self.assertLessEqual(abs(zero).max(),16)
        self.assertTrue(np.array_equal(zero,features(np.zeros(32000,dtype=np.int16))))

    def test_signal_gain_and_duration_contract(self):
        signal=(1000*np.sin(2*np.pi*440*np.arange(32000)/16000)).astype(np.int16)
        a=features(signal);b=features(signal*2)
        self.assertGreater(np.linalg.norm(a-features(np.zeros(32000,dtype=np.int16))),1)
        self.assertLess(np.max(np.abs(a[:240]-b[:240])),.1)
        for invalid in (np.zeros(32001,dtype=np.int16),np.zeros(32000),np.array([1,2],dtype=np.int16)):
            with self.assertRaises(ValueError):features(invalid)

    def test_source_split_is_deterministic_and_not_augmentation_based(self):
        values=[split_for(f'utterance-{i}') for i in range(1000)]
        self.assertEqual(set(values),{'train','validation','test'})
        self.assertEqual(values,[split_for(f'utterance-{i}') for i in range(1000)])
        self.assertTrue(50<values.count('test')<200)
        with self.assertRaises(ValueError):split_for('')
