"""出力channel別の量子化とBatchNorm foldを数値契約で固定する。"""
import unittest
import numpy as np
from arcade_ai_pet import dscnn_quantize as quant


class DscnnQuantizeTests(unittest.TestCase):
    def test_channel_scales_and_q5_bias_are_independent_and_integer(self):
        weights=np.array([[0,0],[.5,-1],[2,-4]],dtype=np.float64);bias=np.array([.5,-.25,3.])
        actual=quant.quantize_layer(weights,bias)
        np.testing.assert_array_equal(actual['weight'],[[0,0],[32,-64],[32,-64]])
        np.testing.assert_array_equal(actual['shift'],[16,6,4]);np.testing.assert_array_equal(actual['bias'],[1048576,-512,1536])
        self.assertEqual(actual['weight'].dtype,np.int8);self.assertEqual(actual['bias'].dtype,np.int32);self.assertEqual(actual['shift'].dtype,np.int32)

    def test_rounding_ties_and_bias_saturation_are_explicit(self):
        w=np.array([[1,.5/64,1.5/64,2.5/64,-1.5/64]],dtype=np.float32)
        r=quant.quantize_layer(w,np.array([2.5/(32*64)],dtype=np.float32))
        np.testing.assert_array_equal(r['weight'],[[64,0,2,2,-2]]);np.testing.assert_array_equal(r['bias'],[2])
        r=quant.quantize_layer(np.zeros((2,1)),np.array([1e6,-1e6]))
        np.testing.assert_array_equal(r['bias'],[2**31-1,-2**31])

    def test_fold_matches_separate_batch_normalization_without_mutation(self):
        rng=np.random.default_rng(21);weights=rng.normal(size=(3,2,1,1));bias=rng.normal(size=3);mean=rng.normal(size=3);var=np.array([0.,.2,3.]);gamma=rng.normal(size=3);beta=rng.normal(size=3)
        before=[v.copy() for v in (weights,bias,mean,var,gamma,beta)]
        fw,fb=quant.fold_batch_norm(weights,bias,mean,var,gamma,beta,eps=.01)
        values=rng.normal(size=(5,2));raw=values@weights[:,:,0,0].T+bias
        expected=(raw-mean)/np.sqrt(var+.01)*gamma+beta
        np.testing.assert_allclose(values@fw[:,:,0,0].T+fb,expected,rtol=1e-12,atol=1e-12)
        for value,original in zip((weights,bias,mean,var,gamma,beta),before,strict=True):np.testing.assert_array_equal(value,original)

    def test_bad_shapes_nonfinite_or_unbounded_parameters_are_refused(self):
        for w,b in ((np.zeros((1,)),np.zeros(1)),(np.zeros((2,3)),np.zeros(3)),(np.array([[np.nan]]),np.zeros(1)),(np.array([[1e30]]),np.zeros(1)),(np.array([[1]],dtype=object),np.zeros(1))):
            with self.assertRaises(ValueError):quant.quantize_layer(w,b)
        args=[np.ones((1,1,1,1)),np.zeros(1),np.zeros(1),np.ones(1),np.ones(1),np.zeros(1)]
        for eps in (0,-1,np.inf,np.nan):
            with self.assertRaises(ValueError):quant.fold_batch_norm(*args,eps=eps)
        args[3][0]=-1
        with self.assertRaises(ValueError):quant.fold_batch_norm(*args,eps=.01)


if __name__=='__main__':unittest.main()
