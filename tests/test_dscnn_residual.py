"""固定prefixの初期化/accumulator/整数headを独立した計算と照合する。"""
import unittest
import numpy as np
from arcade_ai_pet import dscnn_residual as residual
from arcade_ai_pet import dscnn_model as model
from test_dscnn_model import fixture as base_fixture


def reference_accumulator(vector,layers):
    current=np.rint(vector[:2048]*32).astype(np.int64).reshape(1,64,32)
    for layer,stride,groups in zip(layers[:5],(2,1,1,2,1),(1,8,1,16,1),strict=True):
        weight=layer['weight'];channels,depth,k,_=weight.shape;_,h,w=current.shape
        out=np.zeros((channels,(h+stride-1)//stride,(w+stride-1)//stride),dtype=np.int64)
        for oc in range(channels):
            for row in range(out.shape[1]):
                for col in range(out.shape[2]):
                    value=int(layer['bias'][oc])
                    for ic in range(depth):
                        channel=oc if groups!=1 else ic
                        for kr in range(k):
                            for kc in range(k):
                                y=row*stride+kr-k//2;x=col*stride+kc-k//2
                                if 0<=y<h and 0<=x<w:value+=int(current[channel,y,x])*int(weight[oc,ic,kr,kc])
                    out[oc,row,col]=min(192,max(0,value//2**int(layer['shift'][oc])))
        current=out
    joined=[*map(int,current.ravel()),*map(int,np.rint(vector[2048:2113]*32))]
    return np.array([sum(v*int(w) for v,w in zip(joined,row,strict=True))+int(b) for row,b in zip(layers[-1]['weight'],layers[-1]['bias'],strict=True)],dtype=np.int64)


class DscnnResidualTests(unittest.TestCase):
    def test_zero_auxiliary_keeps_old_scores_and_does_not_alias_base(self):
        base=base_fixture();before=residual.frozen_digest(base);new=residual.make_layers(base)
        residual.assert_frozen(base,new)
        values=np.zeros((2,2145),dtype=np.float32);values[:,2112]=2;values[:,2113:]=-3
        np.testing.assert_array_equal(model.scores(values,new,envelope=True),model.scores(values[:,:2113],base))
        self.assertEqual(model.weights_bytes(new,envelope=True)+20,23635)
        new[0]['weight'][0,0,0,0]+=1
        self.assertEqual(residual.frozen_digest(base),before)
        with self.assertRaises(ValueError):residual.assert_frozen(base,new)

    def test_prefix_accumulators_match_scalar_and_integer_head_matches_full_cnn(self):
        rng=np.random.default_rng(312);values=rng.uniform(-1,1,(2,2145)).astype(np.float32)
        values[:,2048:2112]=-abs(values[:,2048:2112]);values[:,2112]=2
        base=base_fixture();acc=residual.prefix_accumulators(values,base)
        for v,a in zip(values,acc,strict=True):np.testing.assert_array_equal(a,reference_accumulator(v,base))
        aux=rng.integers(-8,9,(7,32),dtype=np.int8);new=residual.make_layers(base,aux)
        means=np.rint(values[:,2113:]*32).astype(np.int16)
        predicted=residual.integer_head(acc,means,aux,base[-1]['shift'])
        np.testing.assert_array_equal(predicted,model.scores(values,new,envelope=True))

    def test_signed_scaling_and_int32_saturation_are_exact(self):
        acc=np.array([[2**40,-2**40,15,-15,15,-15,0]],dtype=np.int64)
        shifts=np.array([0,0,8,8,3,3,16],dtype=np.int32)
        result=residual.integer_head(acc,np.zeros((1,32),dtype=np.int16),np.zeros((7,32),dtype=np.int8),shifts)
        np.testing.assert_array_equal(result[0],[2**31-1,-2**31,1,-2,60,-60,0])

    def test_fixed_shift_quantizer_has_ties_even_clipping_and_no_mutation(self):
        shifts=np.arange(7,dtype=np.int32);values=np.zeros((7,32),dtype=np.float64)
        values[:,:5]=np.array([.5,1.5,-2.5,1000000,-1000000])[None,:]/np.exp2(shifts)[:,None]
        before=values.copy();actual=residual.quantize_auxiliary(values,shifts)
        np.testing.assert_array_equal(actual[:,:5],np.tile([0,2,-2,127,-127],(7,1)))
        np.testing.assert_array_equal(values,before);self.assertEqual(actual.dtype,np.int8)

    def test_only_auxiliary_may_change_in_frozen_layers(self):
        base=base_fixture();aux=np.ones((7,32),dtype=np.int8);residual.assert_frozen(base,residual.make_layers(base,aux))
        for field in ('weight','bias','shift'):
            new=residual.make_layers(base,aux);new[-1][field].flat[0]+=1
            with self.assertRaises(ValueError):residual.assert_frozen(base,new)
        with self.assertRaises(ValueError):residual.make_layers(base,np.zeros((7,32),dtype=float))

    def test_bad_accumulators_shapes_values_and_legacy_inputs_refuse(self):
        acc=np.zeros((1,7),dtype=np.int64);means=np.zeros((1,32),dtype=np.int16);aux=np.zeros((7,32),dtype=np.int8);shift=np.zeros(7,dtype=np.int32)
        for args in ((acc.astype(float),means,aux,shift),(acc,np.full((1,32),513,dtype=np.int16),aux,shift),(np.full((1,7),2**40+1,dtype=np.int64),means,aux,shift),(acc,means,aux,np.full(7,17,dtype=np.int32))):
            with self.assertRaises(ValueError):residual.integer_head(*args)
        with self.assertRaises(ValueError):residual.prefix_accumulators(np.zeros((1,2113),dtype=np.float32),base_fixture())
        with self.assertRaises(ValueError):residual.quantize_auxiliary(np.full((7,32),np.nan),shift)

if __name__=='__main__':unittest.main()
