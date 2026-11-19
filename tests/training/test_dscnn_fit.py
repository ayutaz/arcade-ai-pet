"""小さなCPU fixtureでdeterminismと期限/step guardを先行検証する。"""
import unittest
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import dscnn_training as trainer
from arcade_ai_pet.dscnn_model import scores,weights_bytes


def fixture():
    base=np.zeros((7,2113),dtype=np.float32)
    for label in range(7):
        image=base[label,:2048].reshape(64,32);image[4+label*8:8+label*8,2+label*4:6+label*4]=3
        base[label,-1]=label/2
    return np.tile(base,(4,1)),np.tile(np.arange(7),4),np.tile(base,(16,1)),np.tile(np.arange(7),16),np.repeat(np.arange(16),7)


class DscnnFitTests(unittest.TestCase):
    def test_two_small_cpu_fits_are_deterministic_and_export_fixed_integer_layers(self):
        x,y,vx,vy,groups=fixture();before=x.copy()
        a=trainer.fit(x,y,vx,vy,groups,epochs=2,batch_size=14,max_steps=4,max_seconds=60)
        b=trainer.fit(x,y,vx,vy,groups,epochs=2,batch_size=14,max_steps=4,max_seconds=60)
        np.testing.assert_array_equal(scores(vx,a['layers']),scores(vx,b['layers']))
        self.assertEqual(a['training_steps'],4);self.assertEqual(a['selected_epoch'],b['selected_epoch'])
        self.assertEqual(weights_bytes(a['layers'])+20,23411);self.assertLessEqual(a['checkpoint_candidates_evaluated'],16)
        np.testing.assert_array_equal(x,before)

    def test_step_budget_and_foreign_device_refuse_before_model_allocation(self):
        args=fixture()
        for kwargs in ({'epochs':3,'batch_size':7,'max_steps':11},{'device':'cuda'},{'max_seconds':float('nan')},{'epochs':1001}):
            with self.subTest(kwargs=kwargs),patch.object(trainer,'build_model',side_effect=AssertionError('model allocated')):
                with self.assertRaises(ValueError):trainer.fit(*args,**kwargs)

    def test_deadline_stops_instead_of_returning_a_partial_candidate(self):
        args=fixture()
        with patch.object(trainer,'monotonic',side_effect=[0,2]):
            with self.assertRaises(TimeoutError):trainer.fit(*args,epochs=1,batch_size=14,max_seconds=1)


if __name__=='__main__':unittest.main()
