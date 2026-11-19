"""固定整数重みの丸めなしshadowを別のscalar実装と比較する。"""
import math
import unittest
import numpy as np
from arcade_ai_pet import dscnn_shadow as shadow
from test_dscnn_model import fixture


def reference(vector,layers):
    current=vector[:2048].astype(np.float64).reshape(1,64,32)
    for layer,stride,groups in zip(layers[:5],(2,1,1,2,1),(1,8,1,16,1),strict=True):
        w=layer['weight'];ocount,depth,k,_=w.shape;_,height,width=current.shape
        out=np.zeros((ocount,(height+stride-1)//stride,(width+stride-1)//stride))
        for oc in range(ocount):
            scale=2**int(layer['shift'][oc])
            for row in range(out.shape[1]):
                for col in range(out.shape[2]):
                    terms=[int(layer['bias'][oc])/(32*scale)]
                    for ic in range(depth):
                        channel=oc if groups!=1 else ic
                        for kr in range(k):
                            for kc in range(k):
                                y=row*stride+kr-k//2;x=col*stride+kc-k//2
                                if 0<=y<height and 0<=x<width:terms.append(float(current[channel,y,x])*int(w[oc,ic,kr,kc])/scale)
                    out[oc,row,col]=min(6,max(0,math.fsum(terms)))
        current=out
    joined=[*map(float,current.ravel()),*map(float,vector[2048:])]
    return np.array([1024*math.fsum([int(layers[-1]['bias'][i])/(32*2**int(layers[-1]['shift'][i])),
                    *(v*int(w)/2**int(layers[-1]['shift'][i]) for v,w in zip(joined,layers[-1]['weight'][i],strict=True))]) for i in range(7)])


class DscnnShadowTests(unittest.TestCase):
    def test_all_layers_match_independent_real_arithmetic_without_mutation(self):
        rng=np.random.default_rng(91);values=rng.uniform(-.5,.5,(1,2113)).astype(np.float32)
        values[:,2048:2112]=-abs(values[:,2048:2112]);values[:,-1]=2
        layers=fixture();before=values.copy();weights=[{k:v.copy() for k,v in x.items()} for x in layers]
        actual=shadow.scores(values,layers)
        self.assertEqual(actual.dtype,np.float64);self.assertEqual(actual.shape,(1,7))
        np.testing.assert_allclose(actual[0],reference(values[0],layers),rtol=1e-12,atol=1e-9)
        np.testing.assert_array_equal(values,before)
        for a,b in zip(layers,weights):
            for key in a:np.testing.assert_array_equal(a[key],b[key])

    def test_q5_bias_scale_relu6_nchw_head_and_unrounded_tail(self):
        layers=fixture()
        for layer in layers:layer['weight'][:]=0;layer['bias'][:]=0;layer['shift'][:]=0
        layers[4]['bias'][:]=32*13;layers[4]['shift'][:]=0
        layers[-1]['weight'][0,0]=1
        layers[-1]['weight'][1,3072]=1
        layers[-1]['weight'][2,3136]=1
        layers[-1]['bias'][3]=-32;layers[-1]['shift'][3]=1
        values=np.zeros((1,2113),dtype=np.float32);values[0,2048]=-1/64;values[0,-1]=2
        np.testing.assert_array_equal(shadow.scores(values,layers)[0],[6144,-16,2048,-512,0,0,0])

    def test_real_head_is_not_silently_saturated_to_integer_output(self):
        layers=fixture()
        for layer in layers:layer['weight'][:]=0;layer['bias'][:]=0;layer['shift'][:]=0
        layers[-1]['bias'][0]=2**31-1;layers[-1]['bias'][1]=-2**31
        expected=np.array([(2**31-1)*32,(-2**31)*32,0,0,0,0,0],dtype=np.float64)
        np.testing.assert_array_equal(shadow.scores(np.zeros((1,2113),dtype=np.float32),layers)[0],expected)

    def test_batch_equivalence_and_existing_input_layer_contracts(self):
        layers=fixture();values=np.zeros((33,2113),dtype=np.float32);values[:,-1]=np.linspace(0,4,33)
        actual=shadow.scores(values,layers)
        np.testing.assert_allclose(actual,np.vstack([shadow.scores(v[None],layers)[0] for v in values]),rtol=1e-12,atol=1e-9)
        for bad in (np.zeros((1,261),dtype=np.float32),np.zeros((1,2113)),np.full((1,2113),np.nan,dtype=np.float32)):
            with self.assertRaises(ValueError):shadow.scores(bad,layers)
        with self.assertRaises(ValueError):shadow.scores(values,layers[:-1])


if __name__=='__main__':unittest.main()
