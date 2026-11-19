"""先行契約: 小型headのTorch構成/export/CPU学習と上限を保証する。"""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from arcade_ai_pet import dscnn_training as trainer
from arcade_ai_pet.dscnn_model import scores,weights_bytes
from test_envelope_training import fixture

MODE=dict(envelope=True,time_pool=True)


class TimePoolTrainingTests(unittest.TestCase):
    def test_real_architecture_capacity_and_explicit_export_mode(self):
        net=trainer.build_model(**MODE).eval()
        self.assertEqual(net.head.in_features,865);self.assertEqual(sum(p.numel() for p in net.parameters()),7006)
        with torch.no_grad():self.assertEqual(tuple(net(torch.zeros((2,2145))).shape),(2,7))
        layers=trainer.quantize(net,**MODE)
        self.assertEqual(layers[-1]['weight'].shape,(7,865));self.assertEqual(weights_bytes(layers,**MODE)+20,7507)
        for options in ({},{'envelope':True}):
            with self.assertRaises(ValueError):trainer.quantize(net,**options)
        with self.assertRaises(ValueError):trainer.quantize(trainer.build_model(envelope=True).eval(),**MODE)

    def test_forward_pools_time_in_float_and_keeps_all_signed_tail_values(self):
        net=trainer.build_model(**MODE).eval();image=(torch.arange(2*24*16*8).reshape(2,24,16,8)%97).float()/16
        values=torch.zeros((2,2145));values[:,2048:2112]=-.5;values[:,2112]=2;values[:,2113:]=-.25
        captured=[]
        def capture(module,args):captured.append(args[0].detach().clone())
        a=net.blocks[-1].register_forward_hook(lambda module,args,output:image)
        b=net.head.register_forward_pre_hook(capture)
        try:
            with torch.no_grad():net(values)
        finally:a.remove();b.remove()
        pooled=torch.stack([image[:,:,i*4:(i+1)*4,:].sum(2)/4 for i in range(4)],dim=2)
        self.assertTrue(torch.equal(captured[0][:,:768],pooled.flatten(1)))
        self.assertTrue(torch.equal(captured[0][:,768:],values[:,2048:]))
        self.assertTrue(bool((pooled*32 != torch.floor(pooled*32)).any()))

    def test_export_requires_eval_and_preserves_float_state(self):
        net=trainer.build_model(**MODE)
        with self.assertRaises(ValueError):trainer.quantize(net,**MODE)
        net.eval();before={k:v.clone() for k,v in net.state_dict().items()};trainer.quantize(net,**MODE)
        for key,value in net.state_dict().items():self.assertTrue(torch.equal(value,before[key]))

    def test_two_cpu_fits_repeat_with_integer_checkpoint_selection(self):
        args=fixture();before=args[0].copy();options=dict(epochs=2,batch_size=14,max_steps=4,max_seconds=60,**MODE)
        a=trainer.fit(*args,**options);b=trainer.fit(*args,**options)
        np.testing.assert_array_equal(scores(args[2],a['layers'],**MODE),scores(args[2],b['layers'],**MODE))
        self.assertEqual(a['training_steps'],4);self.assertEqual(a['selected_epoch'],b['selected_epoch'])
        self.assertEqual(weights_bytes(a['layers'],**MODE)+20,7507);np.testing.assert_array_equal(args[0],before)

    def test_budget_deadline_and_mode_refuse_without_partial_candidate(self):
        args=fixture()
        for options in ({'time_pool':True},{'envelope':True,'time_pool':1},{**MODE,'max_steps':1}):
            with patch.object(trainer,'build_model',side_effect=AssertionError('allocation reached')):
                with self.assertRaises(ValueError):trainer.fit(*args,**options)
        with patch.object(trainer,'monotonic',side_effect=[0,2]):
            with self.assertRaises(TimeoutError):trainer.fit(*args,epochs=1,max_seconds=1,**MODE)

if __name__=='__main__':unittest.main()
