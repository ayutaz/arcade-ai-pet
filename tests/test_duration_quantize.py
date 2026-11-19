"""261入力だけを追加し、旧量子化器の上限を解除しない。"""
import unittest
import numpy as np
from arcade_ai_pet.duration_quantize import quantize_layer
from arcade_ai_pet.quantize import quantize_layer as previous,dense


class DurationQuantizeTests(unittest.TestCase):
    def test_zero_extra_column_preserves_previous_quantization_exactly(self):
        rng=np.random.default_rng(41);w=rng.normal(0,.05,(14,260));b=rng.normal(0,.02,14)
        old=previous(w,b);new=quantize_layer(np.column_stack((w,np.zeros(14))),b)
        np.testing.assert_array_equal(new['weight'][:,:260],old['weight'])
        np.testing.assert_array_equal(new['bias'],old['bias']);self.assertEqual(new['shift'],old['shift'])

    def test_added_column_participates_in_scaling_and_integer_inference(self):
        w=np.zeros((2,261));w[0,-1]=.5;w[1,0]=.25;b=np.array([.1,-.1])
        layer=quantize_layer(w,b);x=np.zeros(261);x[-1]=2;x[0]=1
        actual=dense(np.rint(x*1024).astype(np.int32),layer)/1024
        np.testing.assert_allclose(actual,w@x+b,atol=.002)
        self.assertEqual(layer['weight'].dtype,np.int8)
        with self.assertRaises(ValueError):previous(w,b)

    def test_new_width_and_numeric_budgets_remain_bounded(self):
        for w,b in ((np.zeros((1,260)),np.zeros(1)),(np.zeros((1,262)),np.zeros(1)),(np.zeros((91,261)),np.zeros(91)),(np.full((1,261),np.nan),np.zeros(1)),(np.zeros((1,261)),np.array([1e20]))):
            with self.assertRaises(ValueError):quantize_layer(w,b)
        first=quantize_layer(np.zeros((90,261)),np.zeros(90));last=previous(np.zeros((7,90)),np.zeros(7))
        self.assertEqual(sum(l['weight'].nbytes+l['bias'].nbytes+4 for l in (first,last)),24516)


if __name__=='__main__':unittest.main()
