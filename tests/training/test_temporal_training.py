"""小型fixtureで実学習し、量子化後の分類と選定を点検。"""
import unittest
import numpy as np
from arcade_ai_pet import temporal_training as training
from arcade_ai_pet.temporal_model import scores,weights_bytes


class TemporalTrainingTests(unittest.TestCase):
    def test_tiny_seven_class_problem_learns_with_integer_validation(self):
        y=np.tile(np.arange(7),16);x=np.zeros((len(y),260),dtype=np.float32)
        for index,label in enumerate(y):x[index,np.arange(20)*12+label]=4
        groups=np.repeat(np.arange(16),7)
        result=training.fit(x,y,x,y,groups,epochs=40,batch_size=32)
        self.assertGreaterEqual(float(np.mean(scores(x,result['layers']).argmax(1)==y)),.95)
        self.assertEqual(weights_bytes(result['layers']),18280)
        self.assertTrue(result['accepted_all_domains']);self.assertLessEqual(result['checkpoint_candidates_evaluated'],16)
        self.assertEqual(result['selection_metric'],'calibrated_worst_development_domain_profile')

    def test_float_and_integer_time_pool_layout_agree_on_exact_fixture(self):
        import torch
        model=training.build_model()
        with torch.no_grad():
            for layer in [*model.convs,model.fc,model.head]:
                layer.weight.zero_();layer.bias.zero_()
                if layer.weight.ndim==3:
                    for out in range(layer.weight.shape[0]):layer.weight[out,out%layer.weight.shape[1],out%layer.weight.shape[2]]=1
                else:
                    for out in range(layer.weight.shape[0]):layer.weight[out,out%layer.weight.shape[1]]=1
        x=np.random.default_rng(5).integers(0,32,(2,260)).astype(np.float32)/4
        from arcade_ai_pet.temporal_model import sequence
        with torch.no_grad():floating=model(torch.tensor(sequence(x).transpose(0,2,1).copy())).numpy()
        integer=scores(x,training.quantize(model))/1024
        np.testing.assert_array_equal(integer,floating)

    def test_unavailable_or_unknown_device_is_rejected_explicitly(self):
        import torch
        from unittest.mock import patch
        y=np.tile(np.arange(7),16);x=np.zeros((len(y),260),dtype=np.float32);g=np.repeat(np.arange(16),7)
        with patch.object(torch.backends.mps,'is_available',return_value=False):
            with self.assertRaises(ValueError):training.fit(x,y,x,y,g,device='mps')
        for device in ('cuda',False):
            with self.assertRaises(ValueError):training.fit(x,y,x,y,g,device=device)

    def test_training_bounds_are_checked_before_allocation(self):
        y=np.tile(np.arange(7),16);x=np.zeros((len(y),260),dtype=np.float32);g=np.repeat(np.arange(16),7)
        for kwargs in (dict(epochs=1001),dict(batch_size=0),dict(batch_size=True)):
            with self.assertRaises(ValueError):training.fit(x,y,x,y,g,**kwargs)


if __name__=='__main__':unittest.main()
