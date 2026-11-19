"""時間情報を加え、既存260特徴と凍結済みコードを維持する契約。"""
import unittest
import numpy as np
from arcade_ai_pet import duration_frontend as duration
from arcade_ai_pet.channel_frontend import features as previous, active_span
from arcade_ai_pet.features import WINDOW, HOP, FRAMES


def tone(length, start=3200):
    pcm=np.zeros(32000,dtype=np.int16)
    pcm[start:start+length]=np.rint(3000*np.sin(2*np.pi*440*np.arange(length)/16000)).astype(np.int16)
    return pcm


class DurationFrontendTests(unittest.TestCase):
    def test_all_existing_features_are_preserved_and_one_bounded_duration_is_appended(self):
        pcm=tone(8000);actual=duration.features(pcm)
        self.assertEqual(actual.shape,(261,));self.assertEqual(actual.dtype,np.float32)
        np.testing.assert_array_equal(actual[:260],previous(pcm))
        x=pcm.astype(float)/abs(pcm.astype(float)).max()
        first,last=active_span(np.mean(np.lib.stride_tricks.sliding_window_view(x,WINDOW)[::HOP]**2,axis=1))
        self.assertAlmostEqual(float(actual[-1]),4*(last-first)/FRAMES,places=6)

    def test_duration_increases_with_longer_speech_but_not_an_untruncated_time_shift(self):
        short=duration.features(tone(4800));long=duration.features(tone(16000))
        self.assertLess(short[-1],long[-1]);self.assertLessEqual(long[-1],4)
        self.assertEqual(short[-1],duration.features(tone(4800,start=4480))[-1])

    def test_silence_and_subthreshold_inputs_stay_zero_including_the_new_value(self):
        for pcm in (np.zeros(32000,dtype=np.int16),np.full(32000,31,dtype=np.int16)):
            np.testing.assert_array_equal(duration.features(pcm),np.zeros(261,dtype=np.float32))

    def test_wrong_pcm_type_or_shape_is_rejected(self):
        for bad in (np.zeros(32000),np.zeros(31999,dtype=np.int16),np.zeros((1,32000),dtype=np.int16)):
            with self.assertRaises(ValueError):duration.features(bad)


if __name__=='__main__':unittest.main()
