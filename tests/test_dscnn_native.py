"""先行契約: 固定6層をC整数演算へ移し、Python参照と同じ出力を得る。"""
import ctypes as C
import unittest
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import dscnn_native as native
from arcade_ai_pet.dscnn_model import scores
from arcade_ai_pet.class_rejection import decisions
from test_dscnn_model import fixture,reference


def vectors(rows=3):
    values=np.random.default_rng(190).uniform(-16,16,(rows,2113)).astype(np.float32)
    values[:,2048:2112]=-abs(values[:,2048:2112]);values[:,2112]=np.linspace(0,4,rows)
    values[0,:8]=np.array([.5,-.5,1.5,-1.5,2.5,-2.5,3.5,-3.5])/32
    return values


class DscnnNativeTests(unittest.TestCase):
    def raw(self,engine,values,out,work=None,bytes_=None,**kw):
        return engine.lib.pet_dscnn_predict(kw.get('input',values.ctypes.data),kw.get('count',2113),
            kw.get('layers',engine.layer_table),kw.get('layer_count',6),kw.get('output',out.ctypes.data),
            kw.get('output_count',7),engine.scratch if work is None else work,
            engine.scratch_bytes if bytes_ is None else bytes_)

    def test_all_logits_match_python_and_scalar_reference_including_rounding_ties(self):
        layers=fixture();values=vectors(2);before=values.copy()
        with native.DscnnNativeModel(layers) as engine:
            result=engine.scores(values)
            np.testing.assert_array_equal(result,scores(values,layers))
            for row,out in zip(values,result,strict=True):np.testing.assert_array_equal(out,reference(row,layers))
            self.assertEqual(result.dtype,np.int32);self.assertEqual(result.shape,(2,7))
            self.assertEqual(engine.scratch_bytes,12288)
        np.testing.assert_array_equal(values,before)

    def test_head_shift_extremes_floor_division_and_int32_saturation(self):
        layers=fixture()
        for row in layers:row['weight'][:]=0;row['bias'][:]=0;row['shift'][:]=0
        layers[-1]['bias'][:]=[2**31-1,-2**31,15,-15,17,-17,0]
        layers[-1]['shift'][:]=[0,0,6,6,16,16,5]
        values=np.zeros((1,2113),np.float32)
        with native.DscnnNativeModel(layers) as engine:
            expected=np.array([2**31-1,-2**31,7,-8,0,-1,0],np.int32)
            np.testing.assert_array_equal(engine.scores(values)[0],expected)
            np.testing.assert_array_equal(engine.scores(values),scores(values,layers))

    def test_convolution_bias_extremes_and_shift0_16_match(self):
        layers=fixture()
        for row in layers[:5]:
            row['weight'][:]=127;row['bias'][::2]=2**31-1;row['bias'][1::2]=-2**31
            row['shift'][::2]=16;row['shift'][1::2]=0
        values=vectors(2)
        with native.DscnnNativeModel(layers) as engine:np.testing.assert_array_equal(engine.scores(values),scores(values,layers))

    def test_batches_and_repeated_calls_are_identical_and_model_is_owned_copy(self):
        layers=fixture();values=vectors(35);expected=scores(values,layers)
        with native.DscnnNativeModel(layers) as engine:
            for row in layers:row['weight'][:]=0
            first=engine.scores(values);second=np.vstack([engine.scores(v[None,:])[0] for v in values])
            np.testing.assert_array_equal(first,expected);np.testing.assert_array_equal(first,second)
        with self.assertRaises(ValueError):engine.scores(values)

    def test_wrong_layer_shape_dtype_and_shift_refuse_before_compiler(self):
        for kind in ('shape','dtype','shift','missing'):
            layers=fixture()
            if kind=='shape':layers[-1]['weight']=layers[-1]['weight'][:,:865]
            if kind=='dtype':layers[0]['weight']=layers[0]['weight'].astype(np.int16)
            if kind=='shift':layers[2]['shift'][0]=17
            if kind=='missing':layers=layers[:5]
            with patch('arcade_ai_pet.dscnn_native.subprocess.run',side_effect=AssertionError('compiler reached')):
                with self.assertRaises(ValueError):native.DscnnNativeModel(layers)

    def test_classification_ties_threshold_boundary_negative_labels_and_extreme_margins(self):
        table=np.array([3,4,5,6,7],np.int32)
        with native.DscnnNativeModel(fixture()) as engine:
            for best,gap in ((0,2),(0,3),(4,6),(4,7),(5,1),(6,1)):
                logits=np.zeros(7,np.int32);logits[best]=gap
                self.assertEqual(engine.classify(logits,table),int(decisions(logits[None,:],table)[0]))
            tied=np.zeros(7,np.int32)
            self.assertEqual(engine.classify(tied,np.zeros(5,np.int32)),0)
            extreme=np.full(7,-2**31,np.int32);extreme[0]=2**31-1
            self.assertEqual(engine.classify(extreme,table),0)
            for bad in ([-1,0,0,0,0],[4097,0,0,0,0]):
                values=np.array(bad,np.int32)
                self.assertEqual(engine.lib.pet_dscnn_classify(tied.ctypes.data,7,values.ctypes.data,5),-1)
                with self.assertRaises(ValueError):engine.classify(tied,values)

    def test_invalid_float_feature_counts_range_tail_and_nonfinite_refuse_without_output_changes(self):
        with native.DscnnNativeModel(fixture()) as engine:
            for index,value in ((0,np.nan),(0,np.inf),(0,17),(2048,1),(2112,-1),(2112,5)):
                values=np.zeros(2113,np.float32);values[index]=value;out=np.full(7,123,np.int32)
                self.assertEqual(self.raw(engine,values,out),-1);np.testing.assert_array_equal(out,np.full(7,123,np.int32))
                with self.assertRaises(ValueError):engine.scores(values[None,:])
            for bad in (np.zeros((0,2113),np.float32),np.zeros((1,261),np.float32),np.zeros((1,2113),np.float64)):
                with self.assertRaises(ValueError):engine.scores(bad)

    def test_null_counts_alignment_and_short_workspace_refuse_before_writing_canaries(self):
        with native.DscnnNativeModel(fixture()) as engine:
            values=vectors(1)[0];out=np.full(7,321,np.int32);work=C.create_string_buffer(engine.scratch_bytes+32)
            for options in ({'input':None},{'count':2112},{'layers':None},{'layer_count':5},{'output':None},{'output_count':6}):
                self.assertEqual(self.raw(engine,values,out,**options),-1)
            for address,bytes_ in ((None,engine.scratch_bytes-1),(C.addressof(work)+1,engine.scratch_bytes),(0,engine.scratch_bytes)):
                self.assertEqual(self.raw(engine,values,out,work=address,bytes_=bytes_),-1)
            for field in ('weight_count','bias_count','shift_count'):
                row=engine.layer_table[0];old=getattr(row,field);setattr(row,field,old-1)
                self.assertEqual(self.raw(engine,values,out),-1);setattr(row,field,old)
            old=engine.layers[0]['shift'][0];engine.layers[0]['shift'][0]=17
            self.assertEqual(self.raw(engine,values,out),-1);engine.layers[0]['shift'][0]=old
            np.testing.assert_array_equal(out,np.full(7,321,np.int32))
            C.memset(work,0x6b,len(work));self.assertEqual(self.raw(engine,values,out,work=work),1)
            self.assertEqual(bytes(work)[engine.scratch_bytes:],b'k'*32)

    def test_misaligned_pointers_and_overlapping_buffers_refuse_before_access(self):
        with native.DscnnNativeModel(fixture()) as engine:
            values=vectors(1)[0];before=values.copy();out=np.full(7,456,np.int32)
            for options in ({'input':values.ctypes.data+1},{'output':out.ctypes.data+1},
                            {'layers':C.addressof(engine.layer_table)+1},{'output':values.ctypes.data}):
                self.assertEqual(self.raw(engine,values,out,**options),-1)
            self.assertEqual(self.raw(engine,values,out,work=values.ctypes.data),-1)
            self.assertEqual(self.raw(engine,values,out,work=out.ctypes.data),-1)
            np.testing.assert_array_equal(values,before);np.testing.assert_array_equal(out,np.full(7,456,np.int32))
            raw=np.zeros(2113*4+1,np.uint8);unaligned=np.ndarray((1,2113),np.float32,buffer=raw,offset=1)
            np.testing.assert_array_equal(engine.scores(unaligned),scores(np.zeros((1,2113),np.float32),fixture()))

if __name__=='__main__':unittest.main()
