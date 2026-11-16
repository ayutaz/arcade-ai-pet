import unittest
import numpy as np
from arcade_ai_pet.behavior_training import dataset, fit_behavior, action_for


class BehaviorTrainingTests(unittest.TestCase):
    def test_seeded_supervised_training_is_reproducible_and_quantized(self):
        x,y=dataset(range(8),samples=12);vx,vy=dataset(range(10000,10004),samples=12)
        a=fit_behavior(x,y,vx,vy,epochs=30,seed=71)
        b=fit_behavior(x,y,vx,vy,epochs=30,seed=71)
        for l,r in zip(a['layers'],b['layers'],strict=True):
            np.testing.assert_array_equal(l['weight'],r['weight'])
            np.testing.assert_array_equal(l['bias'],r['bias'])
        self.assertLess(a['weights_bytes'],2048)
        with self.assertRaises(ValueError):dataset([1,1],samples=12)
        with self.assertRaises(ValueError):fit_behavior(x[:,:2],y,vx,vy,epochs=1)
