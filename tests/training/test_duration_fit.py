"""261入力を明示し、旧260入力との取り違えを拒否する実学習契約。"""
import unittest
import numpy as np
from arcade_ai_pet.domain_training import fit,validation_rank
from arcade_ai_pet.training import scores


class DurationFitTests(unittest.TestCase):
    def test_explicit_261_feature_fit_is_deterministic_and_keeps_all_groups(self):
        y=np.tile(np.arange(7),16);x=np.zeros((112,261),dtype=np.float32);x[np.arange(112),y]=4
        x[:,260]=np.linspace(.4,4,112);groups=np.repeat(np.arange(16),7)
        a=fit(x,y,x,y,groups,epochs=120,hidden=14,feature_count=261)
        b=fit(x,y,x,y,groups,epochs=120,hidden=14,feature_count=261)
        self.assertEqual(a['layers'][0]['weight'].shape,(14,261))
        np.testing.assert_array_equal(scores(x,a['layers']),scores(x,b['layers']))
        self.assertTrue(validation_rank(y,scores(x,a['layers']).argmax(1),groups)[0])

    def test_wrong_or_implicit_feature_width_is_rejected(self):
        y=np.tile(np.arange(7),16);x=np.zeros((112,261),dtype=np.float32);groups=np.repeat(np.arange(16),7)
        with self.assertRaises(ValueError):fit(x,y,x,y,groups,epochs=1)
        for invalid in (True,262,261.0):
            with self.assertRaises(ValueError):fit(x,y,x,y,groups,epochs=1,feature_count=invalid)


if __name__=='__main__':unittest.main()
