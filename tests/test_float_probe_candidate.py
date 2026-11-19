"""先行契約: private float stateを同じ整数候補へ結び付け、JSONへ混入させない。"""
from pathlib import Path
import copy,hashlib,json,stat,tempfile,unittest
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import dscnn_candidate as trainer,dscnn_probe as probe
from test_dscnn_candidate import fixture as old_fixture,patches
from test_float_checkpoint_probe import state_fixture
from test_caption_holdout import SEED


def fixture(root):
    domains,old,new,fitted=old_fixture(root);state=state_fixture()
    fitted.update(layers=probe.quantized_state(state),float_state=state,float_state_sha256=probe.state_digest(state),
        float_logits_sha256='0'*64,float_probe_enabled=True,float_probe_device='cpu',float_probe_selected={'fixture_only':True},
        float_probe_checkpoints=[dict(epoch=1,float_state_sha256=probe.state_digest(state),float_logits_sha256='0'*64,comparison={'fixture_only':True})])
    return domains,old,new,fitted


class FloatProbeCandidateTests(unittest.TestCase):
    def test_opt_in_keeps_selection_inputs_and_saves_typed_private_state_bound_to_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted=fixture(root);stack,fit,_,_=patches(old,new,fitted)
            with stack:r=trainer.train_candidate(domains,root/'model',holdout_seed=SEED,float_probe=True)
            self.assertTrue(fit.call_args.kwargs['float_probe'])
            np.testing.assert_array_equal(fit.call_args.args[2],np.concatenate((old['validation']['x'],new['held_validation']['x'])))
            file=root/'model/float-checkpoint.npz';self.assertTrue(file.is_file());self.assertEqual(stat.S_IMODE(file.stat().st_mode),0o600)
            self.assertEqual(r['float_checkpoint_sha256'],hashlib.sha256(file.read_bytes()).hexdigest())
            with np.load(file,allow_pickle=False) as arrays:
                self.assertEqual(set(arrays.files),set(fitted['float_state']))
                saved={k:arrays[k] for k in arrays.files};self.assertEqual(probe.state_digest(saved),r['float_state_sha256'])
            self.assertNotIn('float_state',r);self.assertNotIn('float_state',json.loads((root/'model/validation.json').read_text()))
            self.assertIn('arcade_ai_pet/dscnn_probe.py',r['source_sha256']);self.assertEqual(len(r['domain_profiles']),24)
            self.assertTrue(r['accepted_validation']);self.assertFalse(r['final_evaluated']);self.assertFalse(r['device_tested'])

    def test_changed_float_state_or_false_device_binding_refuses_before_output(self):
        for kind in ('state','device'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();domains,old,new,fitted=fixture(root)
                if kind=='state':fitted['float_state']['head.bias'][0]=.25
                else:fitted['float_probe_device']='mps'
                stack,*_=patches(old,new,fitted)
                with stack:
                    with self.assertRaises(ValueError):trainer.train_candidate(domains,root/'model',holdout_seed=SEED,float_probe=True)
                self.assertFalse((root/'model').exists())

    def test_float_state_cannot_bind_different_integer_weights_even_with_new_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted=fixture(root)
            fitted['float_state']['head.bias'][0]=.25;fitted['float_state_sha256']=probe.state_digest(fitted['float_state'])
            stack,*_=patches(old,new,fitted)
            with stack:
                with self.assertRaises(ValueError):trainer.train_candidate(domains,root/'model',holdout_seed=SEED,float_probe=True)
            self.assertFalse((root/'model').exists())

    def test_float_state_mutation_during_calibration_refuses_before_saving(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted=fixture(root);stack,*_=patches(old,new,fitted);original=trainer.calibrate
            def changed(*args,**kwargs):
                out=original(*args,**kwargs);fitted['float_state']['head.bias'][0]=.25;return out
            with stack,patch.object(trainer,'calibrate',side_effect=changed):
                with self.assertRaises(ValueError):trainer.train_candidate(domains,root/'model',holdout_seed=SEED,float_probe=True)
            self.assertFalse((root/'model').exists())

    def test_bad_flag_or_residual_mixture_refuses_before_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,*_=fixture(root)
            for options in ({'float_probe':1},{'float_probe':'yes'},{'float_probe':True,'frozen_base':probe.quantized_state(state_fixture())}):
                with patch.object(trainer,'load_legacy_arrays',side_effect=AssertionError('builder reached')):
                    with self.assertRaises(ValueError):trainer.train_candidate(domains,root/'model',holdout_seed=SEED,**options)

    def test_control_has_no_float_artifact_and_undeclared_state_is_rejected(self):
        for extra_state in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();domains,old,new,fitted=fixture(root)
                if not extra_state:
                    for key in list(fitted):
                        if key.startswith('float_'):del fitted[key]
                stack,fit,_,_=patches(old,new,fitted)
                with stack:
                    if extra_state:
                        with self.assertRaises(ValueError):trainer.train_candidate(domains,root/'model',holdout_seed=SEED)
                    else:r=trainer.train_candidate(domains,root/'model',holdout_seed=SEED)
                self.assertNotIn('float_probe',fit.call_args.kwargs);self.assertFalse((root/'model/float-checkpoint.npz').exists())
                if not extra_state:self.assertNotIn('float_checkpoint_sha256',r)

    def test_familiar_regression_and_timeout_still_refuse_acceptance_or_output(self):
        for kind in ('regression','timeout'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();domains,old,new,fitted=fixture(root)
                if kind=='regression':
                    mask=new['validation']['y']==0;new['validation']['x'][mask,:7]=0;new['validation']['x'][mask,1]=2
                stack,fit,_,_=patches(old,new,fitted)
                with stack:
                    if kind=='timeout':
                        fit.side_effect=TimeoutError('budget')
                        with self.assertRaises(TimeoutError):trainer.train_candidate(domains,root/'model',holdout_seed=SEED,float_probe=True)
                    else:
                        r=trainer.train_candidate(domains,root/'model',holdout_seed=SEED,float_probe=True)
                        self.assertTrue(r['selection_accepted']);self.assertFalse(r['accepted_validation'])
                if kind=='timeout':self.assertFalse((root/'model').exists())

if __name__=='__main__':unittest.main()
