"""224係数だけを整数演算に合わせて学習し、固定prefixを保全する。"""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from arcade_ai_pet import dscnn_residual_training as trainer
from arcade_ai_pet.dscnn_residual import frozen_digest,assert_frozen,integer_head,quantize_auxiliary
from arcade_ai_pet.dscnn_model import SHAPES,scores


def fixture():
    base=[dict(weight=np.zeros(shape,dtype=np.int8),bias=np.zeros(shape[0],dtype=np.int32),shift=np.zeros(shape[0],dtype=np.int32)) for shape in SHAPES]
    base[-1]['shift'][:]=8
    values=np.zeros((7,2145),dtype=np.float32);values[:,2112]=1
    for label in range(7):values[label,2113+label]=3
    return base,(np.tile(values,(4,1)),np.tile(np.arange(7),4),np.tile(values,(16,1)),np.tile(np.arange(7),16),np.repeat(np.arange(16),7))


class DscnnResidualTrainingTests(unittest.TestCase):
    def test_ste_forward_is_exact_integer_head_with_finite_nonzero_gradient(self):
        shifts=np.arange(7,dtype=np.int32);real=np.tile(np.array([.5,1.5,-2.5,3.25]*8),(7,1))/np.exp2(shifts)[:,None]
        accum=np.tile(np.arange(-3,4,dtype=np.int64),(2,1));means=np.tile(np.arange(-16,16,dtype=np.int16),(2,1))
        weights=torch.nn.Parameter(torch.tensor(real,dtype=torch.float64))
        actual=trainer._ste_logits(torch.tensor(accum,dtype=torch.float64),torch.tensor(means,dtype=torch.float64),weights,torch.tensor(shifts,dtype=torch.float64))
        expected=integer_head(accum,means,quantize_auxiliary(real,shifts),shifts)/1024
        np.testing.assert_array_equal(actual.detach().numpy(),expected)
        actual.sum().backward();self.assertTrue(torch.isfinite(weights.grad).all());self.assertGreater(float(weights.grad.abs().sum()),0)

    def test_two_real_cpu_fits_repeat_update_only_auxiliary_and_preserve_inputs(self):
        base,args=fixture();before=frozen_digest(base);original=args[0].copy()
        options=dict(base_layers=base,epochs=2,batch_size=14,max_steps=4,max_seconds=60)
        a=trainer.fit(*args,**options);b=trainer.fit(*args,**options)
        assert_frozen(base,a['layers']);self.assertEqual(frozen_digest(base),before)
        self.assertEqual(a['trainable_coefficients'],224);self.assertTrue(a['base_prefix_unchanged']);self.assertEqual(a['training_steps'],4)
        self.assertGreater(a['final_auxiliary_nonzero_coefficients'],0)
        np.testing.assert_array_equal(scores(args[2],a['layers'],envelope=True),scores(args[2],b['layers'],envelope=True))
        np.testing.assert_array_equal(args[0],original)

    def test_budget_mode_or_backend_refuse_before_prefix_work(self):
        base,args=fixture()
        for options in ({'max_steps':1},{'device':'mps'},{'envelope':False},{'max_seconds':float('nan')}):
            with patch.object(trainer,'prefix_accumulators',side_effect=AssertionError('prefix reached')):
                with self.assertRaises(ValueError):trainer.fit(*args,base_layers=base,**options)

    def test_deadline_returns_no_partial_candidate(self):
        base,args=fixture()
        with patch.object(trainer,'monotonic',side_effect=[0,2]):
            with self.assertRaises(TimeoutError):trainer.fit(*args,base_layers=base,max_seconds=1)

    def test_base_mutation_during_precompute_is_detected(self):
        base,args=fixture();original=trainer.prefix_accumulators
        def changed(values,layers):
            output=original(values,layers);layers[-1]['bias'][0]+=1;return output
        with patch.object(trainer,'prefix_accumulators',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'frozen base changed'):trainer.fit(*args,base_layers=base,epochs=1,max_seconds=60)

if __name__=='__main__':unittest.main()
