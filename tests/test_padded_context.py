"""同じ2秒/同じ位置で文脈範囲だけを変えるASR比較を先に定義する。"""
import unittest
import numpy as np
from arcade_ai_pet.utterance_stages import padded_context

class PaddedContextTests(unittest.TestCase):
    def test_only_declared_frame_context_is_kept_at_original_offset(self):
        pcm=(np.arange(32000)%1000+1).astype(np.int16);before=pcm.copy()
        for margin,start,end in ((10,11200,17840),(20,9600,19440)):
            r=padded_context(pcm,first=80,last=100,margin=margin)
            self.assertEqual((r['start_sample'],r['end_sample']),(start,end))
            self.assertEqual(r['pcm'].shape,(32000,));self.assertEqual(r['pcm'].dtype,np.int16)
            np.testing.assert_array_equal(r['pcm'][start:end],pcm[start:end])
            self.assertFalse(r['pcm'][:start].any());self.assertFalse(r['pcm'][end:].any())
        np.testing.assert_array_equal(pcm,before)

    def test_edges_use_complete_frame_extent_and_no_gain_or_repositioning(self):
        pcm=np.full(32000,-32768,dtype=np.int16)
        r=padded_context(pcm,first=0,last=198,margin=20)
        self.assertEqual((r['start_sample'],r['end_sample']),(0,31920))
        self.assertTrue(np.all(r['pcm'][:31920]==-32768));self.assertFalse(r['pcm'][31920:].any())

    def test_invalid_frames_margin_and_pcm_refuse(self):
        pcm=np.ones(32000,dtype=np.int16)
        for first,last,margin in ((-1,20,10),(20,20,10),(0,199,10),(True,20,10),(0,20,30)):
            with self.subTest(first=first,last=last,margin=margin),self.assertRaises(ValueError):padded_context(pcm,first=first,last=last,margin=margin)
        for invalid in (pcm[:-1],pcm.astype(float)):
            with self.assertRaises(ValueError):padded_context(invalid,first=0,last=20,margin=10)

if __name__=='__main__':unittest.main()
