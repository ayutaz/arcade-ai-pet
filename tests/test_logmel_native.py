"""先行契約: PCM→C log-melの精度・bank・境界・入力不変。"""
import ctypes as C
import re,unittest
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import logmel_native as native
from arcade_ai_pet.logmel_frontend import features,pool
from arcade_ai_pet.features import _WINDOW,_MEL
from arcade_ai_pet.channel_training import level
from arcade_ai_pet.channel_robustness import condition


def signals():
    rng=np.random.default_rng(117);t=np.arange(32000)/16000
    speech=np.zeros(32000,np.int16);speech[6300:18100]=(1800*np.sin(2*np.pi*740*t[:11800])*(1+.4*np.sin(2*np.pi*9*t[:11800]))).astype(np.int16)
    rows=[np.zeros(32000,np.int16),np.tile(np.array([-32768,32767],np.int16),16000),speech,rng.integers(-900,901,32000,dtype=np.int16)]
    for pos in (0,4095,4096,15999,31919,31999):
        row=np.zeros(32000,np.int16);row[pos]=3000;rows.append(row)
    return rows


class LogmelNativeTests(unittest.TestCase):
    def test_mathematical_tables_preserve_double_precision_and_sparse_filters(self):
        text=native.render_tables();self.assertEqual(text,native.render_tables())
        self.assertIn('static const double pet_logmel_window[400]',text)
        self.assertIn(float(_WINDOW[0]).hex(),text);self.assertNotIn(float(_WINDOW[0]).hex()+'f',text)
        index_text=re.search(r'pet_logmel_index\[257\]\[2\] = (.*);',text).group(1)
        weight_text=re.search(r'pet_logmel_weight\[257\]\[2\] = (.*);',text).group(1)
        indexes=np.array([int(v) for v in re.findall(r'-?\d+',index_text)]).reshape(257,2)
        weights=np.array([float.fromhex(v) for v in re.findall(r'-?0x[0-9a-f.]+p[+-]\d+',weight_text)]).reshape(257,2)
        restored=np.zeros_like(_MEL)
        for i in range(257):
            for band,w in zip(indexes[i],weights[i],strict=True):
                if band>=0:restored[band,i]=w
        np.testing.assert_array_equal(restored,_MEL)

    def test_fractional_pool_matches_reference_for_short_long_and_boundary_spans(self):
        rng=np.random.default_rng(75)
        with native.LogmelNativeFrontend() as engine:
            for frames in (1,2,3,20,21,63,64,65,197,198):
                for columns in (1,3,32):
                    values=rng.normal(size=(frames,columns));before=values.copy()
                    np.testing.assert_allclose(engine.pool(values),pool(values),rtol=0,atol=2e-13)
                    np.testing.assert_array_equal(values,before)
            for bad in (np.zeros((0,1)),np.zeros((199,1)),np.zeros((1,33)),np.array([[np.nan]]),np.array([[1e7]]),[[1.]]):
                with self.assertRaises(ValueError):engine.pool(bad)

    def test_zero_and_peak_threshold_keep_the_original_silence_contract(self):
        with native.LogmelNativeFrontend() as engine:
            for peak in (0,31,32):
                pcm=np.full(32000,peak,np.int16);before=pcm.copy();actual=engine.features(pcm)
                np.testing.assert_allclose(actual,features(pcm),rtol=0,atol=2e-6)
                if peak<32:np.testing.assert_array_equal(actual,np.zeros(2113,np.float32))
                else:self.assertEqual(float(actual[-1]),4)
                np.testing.assert_array_equal(pcm,before)

    def test_context_spectrum_duration_and_q5_match_on_boundaries_and_full_scale(self):
        with native.LogmelNativeFrontend() as engine:
            self.assertLessEqual(engine.scratch_bytes,32768)
            for pcm in signals():
                before=pcm.copy();expected=features(pcm);actual=engine.features(pcm)
                self.assertEqual(actual.dtype,np.float32);self.assertEqual(actual.shape,(2113,))
                np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-6)
                np.testing.assert_array_equal(np.rint(actual*32),np.rint(expected*32))
                np.testing.assert_array_equal(pcm,before)

    def test_banked_features_are_bit_identical_without_changing_inputs(self):
        with native.LogmelNativeFrontend() as engine:
            for pcm in signals():
                before=pcm.copy();np.testing.assert_array_equal(engine.features(pcm),engine.features(pcm,banked=True))
                np.testing.assert_array_equal(pcm,before)

    def test_existing_hp160_level_and_full_extract_match_python_in_both_layouts(self):
        with native.LogmelNativeFrontend() as engine:
            for pcm in signals()[:4]:
                before=pcm.copy();expected_pcm=level(condition(pcm,'hp160'));expected=features(expected_pcm)
                for banked in (False,True):
                    np.testing.assert_array_equal(engine.prepare(pcm,banked=banked),expected_pcm)
                    np.testing.assert_allclose(engine.extract(pcm,banked=banked),expected,rtol=0,atol=2e-6)
                np.testing.assert_array_equal(pcm,before)

    def test_python_input_flags_and_closed_runtime_refuse(self):
        with native.LogmelNativeFrontend() as engine:
            for fn in (engine.features,engine.extract,engine.prepare):
                for bad in (np.zeros(31999,np.int16),np.zeros(32000,np.int32),np.zeros((1,32000),np.int16),[0]*32000):
                    with self.assertRaises(ValueError):fn(bad)
                for flag in (1,'yes',None):
                    with self.assertRaises(ValueError):fn(np.zeros(32000,np.int16),banked=flag)
        with self.assertRaises(ValueError):engine.features(np.zeros(32000,np.int16))
        with self.assertRaises(ValueError):engine.pool(np.ones((1,1)))

    def test_raw_null_alignment_lengths_overlap_and_workspace_canary(self):
        with native.LogmelNativeFrontend() as engine:
            pcm=signals()[2];before=pcm.copy();out=np.full(2113,123,np.float32);work=C.create_string_buffer(engine.scratch_bytes+32)
            fn=engine.lib.pet_logmel_features
            for ip,n,op,count,wp,size in ((None,32000,out.ctypes.data,2113,work,engine.scratch_bytes),
                (pcm.ctypes.data+1,32000,out.ctypes.data,2113,work,engine.scratch_bytes),
                (pcm.ctypes.data,31999,out.ctypes.data,2113,work,engine.scratch_bytes),
                (pcm.ctypes.data,32000,out.ctypes.data,2112,work,engine.scratch_bytes),
                (pcm.ctypes.data,32000,out.ctypes.data+1,2113,work,engine.scratch_bytes),
                (pcm.ctypes.data,32000,out.ctypes.data,2113,C.addressof(work)+1,engine.scratch_bytes),
                (pcm.ctypes.data,32000,out.ctypes.data,2113,work,engine.scratch_bytes-1),
                (pcm.ctypes.data,32000,pcm.ctypes.data,2113,work,engine.scratch_bytes),
                (pcm.ctypes.data,32000,out.ctypes.data,2113,pcm.ctypes.data,engine.scratch_bytes)):
                self.assertEqual(fn(ip,n,op,count,wp,size),-1)
            np.testing.assert_array_equal(out,np.full(2113,123,np.float32));np.testing.assert_array_equal(pcm,before)
            C.memset(work,0x6b,len(work));self.assertEqual(fn(pcm.ctypes.data,32000,out.ctypes.data,2113,work,engine.scratch_bytes),1)
            self.assertEqual(bytes(work)[engine.scratch_bytes:],b'k'*32)

    def test_bad_banks_and_pool_c_contracts_refuse_before_output(self):
        with native.LogmelNativeFrontend() as engine:
            out=np.full(2113,234,np.float32);banks=[np.zeros(4096,np.int16) for _ in range(8)]
            pointers=(C.c_void_p*8)(*[b.ctypes.data for b in banks]);fn=engine.lib.pet_logmel_features_banked
            for ptr in (None,C.addressof(pointers)+1):self.assertEqual(fn(ptr,32000,out.ctypes.data,2113,engine.scratch,engine.scratch_bytes),-1)
            pointers[7]=None;self.assertEqual(fn(pointers,32000,out.ctypes.data,2113,engine.scratch,engine.scratch_bytes),-1)
            pointers[7]=banks[7].ctypes.data+1;self.assertEqual(fn(pointers,32000,out.ctypes.data,2113,engine.scratch,engine.scratch_bytes),-1)
            np.testing.assert_array_equal(out,np.full(2113,234,np.float32))
            values=np.ones((2,3),np.float64);pooled=np.full((64,3),345,np.float64);pool_fn=engine.lib.pet_logmel_pool
            for ip,rows,columns,op,count in ((None,2,3,pooled.ctypes.data,192),(values.ctypes.data,199,3,pooled.ctypes.data,192),
                (values.ctypes.data,2,33,pooled.ctypes.data,192),(values.ctypes.data,2,3,pooled.ctypes.data,191),
                (values.ctypes.data+1,2,3,pooled.ctypes.data,192),(values.ctypes.data,2,3,values.ctypes.data,192)):
                self.assertEqual(pool_fn(ip,rows,columns,op,count),-1)
            values[0,0]=np.nan;self.assertEqual(pool_fn(values.ctypes.data,2,3,pooled.ctypes.data,192),-1)
            np.testing.assert_array_equal(pooled,np.full((64,3),345,np.float64))

    def test_bad_math_tables_refuse_before_compiler(self):
        for name,bad in (('_MEL',np.zeros((31,257))),('_MEL',np.full((32,257),np.nan)),('_WINDOW',np.zeros(399))):
            with patch.object(native,name,bad),patch.object(native.subprocess,'run',side_effect=AssertionError('compiler reached')):
                with self.assertRaises(ValueError):native.LogmelNativeFrontend()

if __name__=='__main__':unittest.main()
