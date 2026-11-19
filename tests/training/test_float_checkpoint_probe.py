"""先行契約: 元float stateの復元と、診断有無で整数選定が変わらないこと。"""
import json,unittest
from unittest.mock import patch
import numpy as np
import torch
from arcade_ai_pet import dscnn_training as trainer,dscnn_probe as probe
from arcade_ai_pet.dscnn_model import scores
from test_dscnn_fit import fixture as old_fixture


def fixture(mode):
    x,y,vx,vy,g=old_fixture()
    if mode.get('envelope'):x=np.pad(x,((0,0),(0,32)));vx=np.pad(vx,((0,0),(0,32)))
    return x,y,vx,vy,g


def assert_layers(test,a,b):
    for x,y in zip(a,b,strict=True):
        for key in ('weight','bias','shift'):np.testing.assert_array_equal(x[key],y[key])


class FloatCheckpointTrainingTests(unittest.TestCase):
    def test_original_float_state_round_trip_all_modes_binds_integer_layers_and_preserves_rng(self):
        for mode in ({},{'envelope':True},{'envelope':True,'time_pool':True}):
            net=trainer.build_model(**mode).eval();state=probe.capture_state(net,**mode)
            digest=probe.state_digest(state,**mode);rng=torch.get_rng_state().clone()
            restored=probe.restore_state(state,**mode)
            self.assertTrue(torch.equal(torch.get_rng_state(),rng));self.assertFalse(restored.training)
            self.assertEqual(digest,probe.state_digest(probe.capture_state(restored,**mode),**mode))
            assert_layers(self,probe.quantized_state(state,**mode),trainer.quantize(net,**mode))
            values=fixture(mode)[2]
            np.testing.assert_array_equal(probe.float_scores(net,values,**mode),probe.float_scores(restored,values,**mode))
            state['head.bias'][0]+=.5
            self.assertEqual(digest,probe.state_digest(probe.capture_state(net,**mode),**mode))

    def test_float_eval_matches_direct_batches_and_leaves_bn_input_and_gradients_unchanged(self):
        net=trainer.build_model().eval();values=np.tile(fixture({})[2],(3,1))[:257].copy();before=values.copy()
        state=probe.state_digest(probe.capture_state(net))
        with torch.no_grad():expected=torch.cat([net(torch.from_numpy(values[i:i+256])) for i in range(0,len(values),256)]).numpy()
        actual=probe.float_scores(net,values)
        np.testing.assert_array_equal(actual,expected);self.assertEqual(actual.dtype,np.float32)
        self.assertEqual(state,probe.state_digest(probe.capture_state(net)));self.assertTrue(all(p.grad is None for p in net.parameters()))
        np.testing.assert_array_equal(values,before)

    def test_bad_mode_input_device_or_nondefault_bn_refuses_before_forward(self):
        net=trainer.build_model().eval();values=fixture({})[2]
        with patch.object(net,'forward',side_effect=AssertionError('forward reached')):
            for bad,options in ((values.astype(float),{}),(values,{'device':'cuda'}),(values,{'envelope':True})):
                with self.assertRaises(ValueError):probe.float_scores(net,bad,**options)
            net.train()
            with self.assertRaises(ValueError):probe.float_scores(net,values)
        net.eval();net.blocks[0][1].eps=1e-4
        with self.assertRaises(ValueError):probe.capture_state(net)

    def test_probe_and_control_fits_select_identical_integer_layers_in_all_modes(self):
        for mode in ({},{'envelope':True},{'envelope':True,'time_pool':True}):
            args=fixture(mode);options=dict(epochs=2,batch_size=14,max_steps=4,max_seconds=60,**mode)
            control=trainer.fit(*args,**options);enabled=trainer.fit(*args,float_probe=True,**options)
            assert_layers(self,control['layers'],enabled['layers']);self.assertEqual(control['selected_epoch'],enabled['selected_epoch'])
            self.assertNotIn('float_state',control);self.assertTrue(enabled['float_probe_enabled'])
            state=enabled['float_state'];self.assertEqual(probe.state_digest(state,**mode),enabled['float_state_sha256'])
            restored=probe.restore_state(state,**mode);floating=probe.float_scores(restored,args[2],**mode)
            self.assertEqual(probe.logits_digest(floating),enabled['float_logits_sha256'])
            self.assertEqual(probe.compare(args[3],floating,scores(args[2],enabled['layers'],**mode),args[4]),enabled['float_probe_selected'])
            self.assertEqual([r['epoch'] for r in enabled['float_probe_checkpoints']],[1,2])
            json.dumps({k:v for k,v in enabled.items() if k not in ('layers','float_state')},allow_nan=False)

    def test_probe_statistics_never_choose_the_checkpoint(self):
        args=fixture({});options=dict(epochs=2,batch_size=14,max_steps=4,max_seconds=60)
        control=trainer.fit(*args,**options)
        with patch.object(probe,'compare',return_value={'not_a_selection_score':10**9}):
            enabled=trainer.fit(*args,float_probe=True,**options)
        self.assertEqual(enabled['selected_epoch'],control['selected_epoch']);assert_layers(self,enabled['layers'],control['layers'])

    def test_bad_flag_step_budget_and_deadline_refuse_before_allocation(self):
        args=fixture({})
        for options in ({'float_probe':'yes'},{'float_probe':1},{'float_probe':True,'max_steps':1}):
            with patch.object(trainer,'build_model',side_effect=AssertionError('allocation reached')):
                with self.assertRaises(ValueError):trainer.fit(*args,**options)
        with patch.object(trainer,'monotonic',side_effect=[0,2]):
            with self.assertRaises(TimeoutError):trainer.fit(*args,float_probe=True,max_seconds=1)

    def test_probe_failure_never_returns_a_partial_candidate(self):
        args=fixture({})
        with patch.object(probe,'float_scores',side_effect=TimeoutError('probe deadline')):
            with self.assertRaises(TimeoutError):trainer.fit(*args,float_probe=True,epochs=1,batch_size=14,max_steps=2,max_seconds=60)

if __name__=='__main__':unittest.main()
