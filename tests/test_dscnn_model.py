"""6層の整数CNNをPython整数による別実装と照合する。"""
import unittest
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import dscnn_model as model

SHAPES=((8,1,3,3),(8,1,3,3),(16,8,1,1),(16,1,3,3),(24,16,1,1),(7,3137))
STRIDES=(2,1,1,2,1)
GROUPS=(1,8,1,16,1)


def fixture():
    rng=np.random.default_rng(771)
    return [dict(weight=rng.integers(-2,3,shape,dtype=np.int8),bias=rng.integers(-30,31,shape[0],dtype=np.int32),shift=rng.integers(0,4,shape[0],dtype=np.int32) if i<5 else np.arange(3,10,dtype=np.int32)) for i,shape in enumerate(SHAPES)]


def reference(vector,layers):
    current=np.asarray([round(float(v)*32) for v in vector[:2048]],dtype=np.int64).reshape(1,64,32)
    for layer,stride,groups in zip(layers[:5],STRIDES,GROUPS,strict=True):
        weight=layer['weight'];out_channels,depth,k,_=weight.shape;_,h,w=current.shape;pad=k//2
        out=np.zeros((out_channels,(h+stride-1)//stride,(w+stride-1)//stride),dtype=np.int64)
        for oc in range(out_channels):
            for row in range(out.shape[1]):
                for col in range(out.shape[2]):
                    total=int(layer['bias'][oc])
                    for ic in range(depth):
                        channel=oc if groups!=1 else ic
                        for kr in range(k):
                            for kc in range(k):
                                y=row*stride+kr-pad;x=col*stride+kc-pad
                                if 0<=y<h and 0<=x<w:total+=int(current[channel,y,x])*int(weight[oc,ic,kr,kc])
                    out[oc,row,col]=min(192,max(0,total//(2**int(layer['shift'][oc]))))
        current=out
    combined=list(map(int,current.ravel()))+[round(float(v)*32) for v in vector[2048:]]
    output=[]
    for oc in range(7):
        total=sum(v*int(w) for v,w in zip(combined,layers[-1]['weight'][oc],strict=True))+int(layers[-1]['bias'][oc])
        shift=int(layers[-1]['shift'][oc])-5
        value=total//(2**shift) if shift>=0 else total*(2**-shift)
        output.append(min(2**31-1,max(-2**31,value)))
    return np.asarray(output,dtype=np.int32)


class DscnnModelTests(unittest.TestCase):
    def test_fixed_architecture_fits_the_numeric_budget_with_channel_shifts(self):
        layers=fixture();self.assertEqual(model.weights_bytes(layers),23391)
        self.assertLessEqual(model.weights_bytes(layers)+20,24576)
        self.assertEqual(model.FEATURE_COUNT,2113)
        self.assertEqual(model.activation_layout()['ping_pong_bytes_estimate'],12288)
        self.assertFalse(model.activation_layout()['target_workspace_measured'])

    def test_full_integer_pipeline_matches_scalar_reference_without_input_mutation(self):
        layers=fixture();rng=np.random.default_rng(49);values=rng.uniform(-1,1,(2,2113)).astype(np.float32)
        values[:,2048:2112]=-abs(values[:,2048:2112]);values[:,-1]=[0,4];values[0,:8]=np.array([.5,-.5,1.5,-1.5,2.5,-2.5,3.5,-3.5])/32
        before=values.copy();actual=model.scores(values,layers)
        self.assertEqual(actual.dtype,np.int32);self.assertEqual(actual.shape,(2,7))
        for row,out in zip(values,actual,strict=True):np.testing.assert_array_equal(out,reference(row,layers))
        np.testing.assert_array_equal(values,before)

    def test_head_extremes_saturate_int32_after_signed_scaling(self):
        layers=fixture()
        for layer in layers:layer['weight'][:]=0;layer['bias'][:]=0;layer['shift'][:]=0
        layers[-1]['bias'][:]=np.array([2**31-1,-2**31,15,-15,1,-1,0],dtype=np.int32)
        values=np.zeros((1,2113),dtype=np.float32)
        expected=np.array([2**31-1,-2**31,480,-480,32,-32,0],dtype=np.int32)
        np.testing.assert_array_equal(model.scores(values,layers)[0],expected)

    def test_wrong_layer_shapes_dtypes_shifts_or_depth_refuse_before_convolution(self):
        for mode in ('shape','float','bias','shift','depth'):
            with self.subTest(mode=mode):
                layers=fixture()
                if mode=='shape':layers[0]['weight']=layers[0]['weight'][:,:,:2]
                elif mode=='float':layers[0]['weight']=layers[0]['weight'].astype(np.float32)
                elif mode=='bias':layers[0]['bias']=layers[0]['bias'].astype(np.int64)
                elif mode=='shift':layers[0]['shift'][0]=17
                else:layers=layers[:-1]
                with patch.object(model.np,'pad',side_effect=AssertionError('allocation reached')):
                    with self.assertRaises(ValueError):model.scores(np.zeros((1,2113),dtype=np.float32),layers)

    def test_wrong_features_legacy_width_and_unphysical_tail_refuse(self):
        layers=fixture()
        invalid=[np.zeros((1,261),dtype=np.float32),np.zeros((1,2113),dtype=np.float64),np.zeros((0,2113),dtype=np.float32)]
        for index,value in ((0,np.nan),(0,17),(2048,1),(2112,-1),(2112,5)):
            row=np.zeros((1,2113),dtype=np.float32);row[0,index]=value;invalid.append(row)
        for row in invalid:
            with self.assertRaises(ValueError):model.scores(row,layers)

    def test_batched_and_individual_scores_are_identical(self):
        layers=fixture();values=np.zeros((35,2113),dtype=np.float32);values[:,-1]=np.linspace(0,4,35)
        actual=model.scores(values,layers)
        expected=np.vstack([model.scores(v[None,:],layers)[0] for v in values])
        np.testing.assert_array_equal(actual,expected)


if __name__=='__main__':unittest.main()
