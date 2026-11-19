"""発話ごとの係数平均だけを除き、時間・energy・入力を保持する。"""
import unittest
import numpy as np
from arcade_ai_pet import cmn_frontend as cmn
from arcade_ai_pet.duration_frontend import features as previous


class CmnFrontendTests(unittest.TestCase):
    def test_coefficient_offsets_cancel_without_changing_energy_duration_or_input(self):
        values=np.random.default_rng(71).uniform(-2,2,261).astype(np.float32);values[-1]=2
        before=values.copy();shifted=values.copy();shifted[:240]=(shifted[:240].reshape(20,12)+np.arange(12,dtype=np.float32)/8).ravel()
        actual=cmn.center(values);other=cmn.center(shifted)
        np.testing.assert_allclose(actual[:240],other[:240],atol=5e-7,rtol=0)
        np.testing.assert_allclose(actual[:240].reshape(20,12).mean(0),0,atol=5e-8)
        np.testing.assert_array_equal(actual[240:],values[240:]);np.testing.assert_array_equal(values,before)
        self.assertFalse(np.shares_memory(actual,values));self.assertEqual(actual.dtype,np.float32)

    def test_batch_normalization_is_per_utterance_and_equals_separate_calls(self):
        rows=np.random.default_rng(9).uniform(-2,2,(3,261)).astype(np.float32);rows[:,-1]=1
        rows[1,:240]+=5;before=rows.copy();actual=cmn.center(rows)
        for i,row in enumerate(rows):np.testing.assert_array_equal(actual[i],cmn.center(row))
        np.testing.assert_allclose(actual[:,:240].reshape(3,20,12).mean(1),0,atol=8e-8)
        np.testing.assert_array_equal(actual[:,240:],rows[:,240:]);np.testing.assert_array_equal(rows,before)

    def test_silence_clipping_and_malformed_values_have_explicit_bounds(self):
        np.testing.assert_array_equal(cmn.center(np.zeros(261,dtype=np.float32)),np.zeros(261,dtype=np.float32))
        x=np.full(261,16,dtype=np.float32);x[-1]=4;x[:12]=-16
        actual=cmn.center(x);self.assertTrue(np.isfinite(actual).all());self.assertLessEqual(abs(actual).max(),16)
        bads=[np.zeros(260,dtype=np.float32),np.zeros(261),np.zeros((0,261),dtype=np.float32),np.zeros((1,1,261),dtype=np.float32),np.full(261,np.nan,dtype=np.float32),np.full(261,17,dtype=np.float32),[0]*261]
        for bad in bads:
            with self.subTest(shape=getattr(bad,'shape',None)),self.assertRaises(ValueError):cmn.center(bad)

    def test_pcm_frontend_uses_existing_context_and_does_not_mutate_pcm(self):
        pcm=np.zeros(32000,dtype=np.int16);pcm[3200:12800]=np.rint(2000*np.sin(np.arange(9600)*.24)).astype(np.int16);before=pcm.copy()
        np.testing.assert_array_equal(cmn.features(pcm),cmn.center(previous(pcm,context=True)))
        np.testing.assert_array_equal(pcm,before)
        np.testing.assert_array_equal(cmn.features(np.zeros(32000,dtype=np.int16)),np.zeros(261,dtype=np.float32))
        for bad in (np.zeros(32000),np.zeros(31999,dtype=np.int16)):
            with self.assertRaises(ValueError):cmn.features(bad)


if __name__=='__main__':unittest.main()
