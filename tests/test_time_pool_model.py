"""先行契約: 時間16→4の整数平均と新headを独立scalarで照合する。"""
import unittest
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import dscnn_model as model
from test_dscnn_model import fixture as old_fixture,STRIDES,GROUPS

MODE=dict(envelope=True,time_pool=True)


def fixture():
    rows=old_fixture();rng=np.random.default_rng(932)
    rows[-1]['weight']=rng.integers(-2,3,(7,865),dtype=np.int8)
    return rows


def reference(vector,layers):
    current=np.asarray([round(float(v)*32) for v in vector[:2048]],dtype=np.int64).reshape(1,64,32)
    for layer,stride,groups in zip(layers[:5],STRIDES,GROUPS,strict=True):
        weight=layer['weight'];channels,depth,k,_=weight.shape;_,height,width=current.shape;pad=k//2
        out=np.zeros((channels,(height+stride-1)//stride,(width+stride-1)//stride),dtype=np.int64)
        for oc in range(channels):
            for row in range(out.shape[1]):
                for col in range(out.shape[2]):
                    total=int(layer['bias'][oc])
                    for ic in range(depth):
                        channel=oc if groups!=1 else ic
                        for kr in range(k):
                            for kc in range(k):
                                y=row*stride+kr-pad;x=col*stride+kc-pad
                                if 0<=y<height and 0<=x<width:total+=int(current[channel,y,x])*int(weight[oc,ic,kr,kc])
                    out[oc,row,col]=min(192,max(0,total//(2**int(layer['shift'][oc]))))
        current=out
    combined=[sum(int(current[c,group*4+t,band]) for t in range(4))//4 for c in range(24) for group in range(4) for band in range(8)]
    combined.extend(round(float(v)*32) for v in vector[2048:])
    result=[]
    for oc in range(7):
        total=sum(v*int(w) for v,w in zip(combined,layers[-1]['weight'][oc],strict=True))+int(layers[-1]['bias'][oc])
        shift=int(layers[-1]['shift'][oc])-5
        value=total//(2**shift) if shift>=0 else total*(2**-shift)
        result.append(min(2**31-1,max(-2**31,value)))
    return np.asarray(result,dtype=np.int32)


class TimePoolModelTests(unittest.TestCase):
    def test_new_shape_capacity_and_explicit_modes_refuse_mixed_heads(self):
        layers=fixture();self.assertEqual(model.weights_bytes(layers,**MODE)+20,7507)
        for options in ({},{'envelope':True},{'time_pool':True},{'envelope':True,'time_pool':1}):
            with self.assertRaises(ValueError):model.validate(layers,**options)
        for width in (3137,3169):
            bad=fixture();bad[-1]['weight']=np.zeros((7,width),dtype=np.int8)
            with self.assertRaises(ValueError):model.validate(bad,**MODE)

    def test_time_axis_floor_and_uint8_bounds_are_independent_of_band_order(self):
        rng=np.random.default_rng(41);values=rng.integers(0,193,(2,24,16,8),dtype=np.uint8);before=values.copy()
        expected=np.array([[[[sum(int(values[n,c,g*4+t,b]) for t in range(4))//4 for b in range(8)] for g in range(4)] for c in range(24)] for n in range(2)],dtype=np.uint8)
        np.testing.assert_array_equal(model._pool_time(values),expected);np.testing.assert_array_equal(values,before)
        for bad in (values.astype(np.int16),values[:,:,:15],np.zeros((0,24,16,8),dtype=np.uint8),np.full((1,24,16,8),193,dtype=np.uint8)):
            with self.assertRaises(ValueError):model._pool_time(bad)

    def test_full_integer_pipeline_matches_scalar_and_preserves_inputs(self):
        rng=np.random.default_rng(41);values=rng.uniform(-1,1,(2,2145)).astype(np.float32)
        values[:,2048:2112]=-abs(values[:,2048:2112]);values[:,2112]=[0,4]
        before=values.copy();layers=fixture();actual=model.scores(values,layers,**MODE)
        for value,result in zip(values,actual,strict=True):np.testing.assert_array_equal(result,reference(value,layers))
        self.assertEqual(actual.dtype,np.int32);np.testing.assert_array_equal(values,before)

    def test_auxiliary_tail_order_and_signed_q5_contribution_survive_pooling(self):
        layers=fixture()
        for row in layers:row['weight'][:]=0;row['bias'][:]=0;row['shift'][:]=0
        layers[-1]['weight'][0,833]=2;layers[-1]['weight'][1,832]=1;layers[-1]['weight'][2,768]=1
        values=np.zeros((1,2145),dtype=np.float32);values[0,2113]=-.25;values[0,2112]=4;values[0,2048]=-.5
        np.testing.assert_array_equal(model.scores(values,layers,**MODE)[0],[-512,4096,-512,0,0,0,0])

    def test_batch_boundary_matches_individual_and_bad_mode_refuses_before_conv(self):
        values=np.zeros((33,2145),dtype=np.float32);values[:,2112]=2;values[:,2113:]=np.linspace(-2,2,33)[:,None]
        layers=fixture();expected=np.vstack([model.scores(v[None],layers,**MODE)[0] for v in values])
        np.testing.assert_array_equal(model.scores(values,layers,**MODE),expected)
        with patch.object(model,'_conv',side_effect=AssertionError('conv reached')):
            for mode in ({'time_pool':True},{'envelope':True,'time_pool':'yes'}):
                with self.assertRaises(ValueError):model.scores(values,layers,**mode)

if __name__=='__main__':unittest.main()
