"""新しい入力/headを実Torchと小型CPU学習で保証する。"""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from arcade_ai_pet import dscnn_training as trainer
from arcade_ai_pet.dscnn_model import scores,weights_bytes
from test_dscnn_fit import fixture as previous_fixture


def fixture():
    x,y,vx,vy,groups=previous_fixture()
    return np.pad(x,((0,0),(0,32))),y,np.pad(vx,((0,0),(0,32))),vy,groups


class EnvelopeTrainingTests(unittest.TestCase):
    def test_real_architecture_and_export_have_the_new_head_and_explicit_mode(self):
        net=trainer.build_model(envelope=True).eval()
        with torch.no_grad():self.assertEqual(tuple(net(torch.zeros((2,2145))).shape),(2,7))
        self.assertEqual(net.head.in_features,3169);self.assertEqual(sum(p.numel() for p in net.parameters()),23134)
        self.assertEqual(weights_bytes(trainer.quantize(net,envelope=True),envelope=True)+20,23635)
        with self.assertRaises(ValueError):net(torch.zeros((2,2113)))
        with self.assertRaises(ValueError):trainer.quantize(net)
        with self.assertRaises(ValueError):trainer.quantize(trainer.build_model().eval(),envelope=True)

    def test_export_requires_eval_and_preserves_new_float_state(self):
        net=trainer.build_model(envelope=True)
        with self.assertRaises(ValueError):trainer.quantize(net,envelope=True)
        net.eval();before={k:v.clone() for k,v in net.state_dict().items()};trainer.quantize(net,envelope=True)
        for key,value in net.state_dict().items():self.assertTrue(torch.equal(value,before[key]))

    def test_two_cpu_fits_are_repeatable_and_use_envelope_integer_selection(self):
        x,y,vx,vy,g=fixture();before=x.copy()
        options=dict(epochs=2,batch_size=14,max_steps=4,max_seconds=60,envelope=True)
        a=trainer.fit(x,y,vx,vy,g,**options);b=trainer.fit(x,y,vx,vy,g,**options)
        np.testing.assert_array_equal(scores(vx,a['layers'],envelope=True),scores(vx,b['layers'],envelope=True))
        self.assertEqual(a['training_steps'],4);self.assertEqual(weights_bytes(a['layers'],envelope=True)+20,23635)
        np.testing.assert_array_equal(x,before)

    def test_budget_invalid_mode_or_legacy_width_refuse_before_allocation(self):
        args=fixture()
        for options in ({'envelope':True,'max_steps':1},{'envelope':'yes'},{'envelope':False}):
            with patch.object(trainer,'build_model',side_effect=AssertionError('allocation reached')):
                with self.assertRaises(ValueError):trainer.fit(*args,**options)

if __name__=='__main__':unittest.main()
