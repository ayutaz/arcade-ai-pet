import unittest
import numpy as np
from arcade_ai_pet.quantize import quantize_layer, dense, predict


class QuantizeTests(unittest.TestCase):
    def test_power_of_two_integer_layer_matches_float_and_relu_bounds(self):
        rng=np.random.default_rng(77)
        w=rng.normal(0,.05,(32,260)).astype(np.float32);b=rng.normal(0,.02,32).astype(np.float32)
        layer=quantize_layer(w,b);x=rng.normal(0,1,260).astype(np.float32)
        actual=dense(np.rint(x*1024).astype(np.int32),layer,relu=True)/1024
        expected=np.clip(w@x+b,0,6)
        self.assertLess(np.max(abs(actual-expected)),.08)
        self.assertTrue(np.all((actual>=0)&(actual<=6)))

    def test_prediction_and_model_bounds(self):
        first=quantize_layer(np.zeros((32,260)),np.zeros(32))
        last=quantize_layer(np.zeros((7,32)),np.array([0,0,0,2,0,0,0]))
        self.assertEqual(predict(np.zeros(260,dtype=np.float32),[first,last]),3)
        for weights,bias in ((np.full((1,1),np.nan),np.zeros(1)),(np.zeros((500,500)),np.zeros(500))):
            with self.assertRaises(ValueError):quantize_layer(weights,bias)
