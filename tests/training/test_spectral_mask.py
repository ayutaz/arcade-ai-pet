"""先行契約: maskは学習batchのみ、評価/末尾/RNG/既定を保持。"""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from arcade_ai_pet import spectral_mask as mask,dscnn_training as trainer,dscnn_probe as probe
from test_dscnn_fit import fixture
from test_float_checkpoint_probe import assert_layers


OPTIONS=dict(epochs=2,batch_size=14,max_steps=4,max_seconds=60)


class SpectralMaskTrainingTests(unittest.TestCase):
    def test_hand_written_union_zeroes_only_time_and_frequency_intervals_and_keeps_tail(self):
        values=torch.arange(2*2113,dtype=torch.float32).reshape(2,2113)/1000-2
        before=values.clone();plans=np.array([[2,3,4,2],[64,0,32,0]],np.int32)
        expected=before.clone();image=expected[0,:2048].reshape(64,32)
        image[2:5,:]=0;image[:,4:6]=0
        result=mask.apply(values,plans)
        self.assertTrue(torch.equal(result,expected));self.assertTrue(torch.equal(values,before))
        self.assertNotEqual(result.data_ptr(),values.data_ptr());self.assertTrue(torch.equal(result[:,2048:],values[:,2048:]))

    def test_invalid_tensors_and_plans_refuse(self):
        valid=torch.ones((1,2113),dtype=torch.float32);plan=np.zeros((1,4),np.int32)
        for values in (valid.double(),valid[:,:2112],valid.numpy(),torch.full_like(valid,float('nan'))):
            with self.assertRaises(ValueError):mask.apply(values,plan)
        with self.assertRaises(ValueError):mask.apply(valid,np.array([[0,8,0,0]],np.int32))

    def test_repeated_fit_is_deterministic_keeps_inputs_and_float_state_binding(self):
        args=fixture();before=[a.copy() for a in args]
        a=trainer.fit(*args,spectral_mask=True,float_probe=True,**OPTIONS)
        b=trainer.fit(*args,spectral_mask=True,float_probe=True,**OPTIONS)
        assert_layers(self,a['layers'],b['layers']);self.assertEqual(a['selected_epoch'],b['selected_epoch'])
        self.assertEqual(a['spectral_mask'],dict(time_max=6,frequency_max=3,fill=0,seed=43,training_only=True))
        self.assertEqual(a['weights_bytes']+20,23411)
        assert_layers(self,probe.quantized_state(a['float_state']),a['layers'])
        for old,new in zip(before,args,strict=True):np.testing.assert_array_equal(old,new)

    def test_mask_off_never_calls_augmentation_and_explicit_false_matches_default(self):
        args=fixture()
        with patch.object(mask,'make_plan',side_effect=AssertionError('mask used')):
            a=trainer.fit(*args,**OPTIONS);b=trainer.fit(*args,spectral_mask=False,**OPTIONS)
        assert_layers(self,a['layers'],b['layers']);self.assertNotIn('spectral_mask',a)

    def test_only_training_batches_are_masked_and_training_order_rng_is_unchanged(self):
        args=fixture();orders=[];original_order=torch.randperm;original_apply=mask.apply
        def order(*a,**kw):
            out=original_order(*a,**kw);orders.append(out.clone());return out
        with patch.object(torch,'randperm',side_effect=order):trainer.fit(*args,**OPTIONS)
        original_orders=orders.copy();orders.clear();seen=[]
        def applied(values,plans):
            seen.append(values.shape);return original_apply(values,plans)
        original_scores=trainer.scores
        def evaluated(values,*a,**kw):
            np.testing.assert_array_equal(values,args[2]);return original_scores(values,*a,**kw)
        with patch.object(torch,'randperm',side_effect=order),patch.object(mask,'apply',side_effect=applied),patch.object(trainer,'scores',side_effect=evaluated):
            trainer.fit(*args,spectral_mask=True,**OPTIONS)
        self.assertEqual(seen,[torch.Size([14,2113])]*4)
        for a,b in zip(original_orders,orders,strict=True):self.assertTrue(torch.equal(a,b))

    def test_bad_flag_incompatible_modes_step_budget_and_deadline_refuse(self):
        args=fixture()
        for options in ({'spectral_mask':1},{'spectral_mask':'yes'},{'spectral_mask':True,'envelope':True},{'spectral_mask':True,'max_steps':1}):
            with patch.object(trainer,'build_model',side_effect=AssertionError('allocation reached')):
                with self.assertRaises(ValueError):trainer.fit(*args,**options)
        with patch.object(trainer,'monotonic',side_effect=[0,2]):
            with self.assertRaises(TimeoutError):trainer.fit(*args,spectral_mask=True,max_seconds=1)
        with patch.object(mask,'apply',side_effect=TimeoutError('augmentation deadline')):
            with self.assertRaises(TimeoutError):trainer.fit(*args,spectral_mask=True,**OPTIONS)

if __name__=='__main__':unittest.main()
