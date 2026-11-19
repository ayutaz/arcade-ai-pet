import unittest
import numpy as np
from arcade_ai_pet.domain_training import fit,validation_rank
from arcade_ai_pet.training import scores


class DomainFitTests(unittest.TestCase):
    def test_deterministic_quantized_fit_keeps_all_sixteen_validation_groups(self):
        y=np.tile(np.arange(7),16);x=np.zeros((112,260),dtype=np.float32);x[np.arange(112),y]=4
        groups=np.repeat(np.arange(16),7)
        a=fit(x,y,x,y,groups,epochs=120,hidden=14);b=fit(x,y,x,y,groups,epochs=120,hidden=14)
        pa=scores(x,a['layers']);pb=scores(x,b['layers']);np.testing.assert_array_equal(pa,pb)
        self.assertTrue(validation_rank(y,pa.argmax(1),groups)[0])
        self.assertEqual(a['selected_epoch'],b['selected_epoch'])
        self.assertLessEqual(a['checkpoint_candidates_evaluated'],16)
        self.assertEqual(a['selection_metric'],'calibrated_worst_development_domain_profile')


if __name__=='__main__':unittest.main()
