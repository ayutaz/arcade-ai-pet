"""固定土台の保全を、候補選定/保存の前提として強制する。"""
from contextlib import ExitStack
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import dscnn_candidate as trainer
from arcade_ai_pet.dscnn_residual import make_layers,frozen_digest
from test_envelope_candidate import fixture as previous_fixture,fake_scores,MODE
from test_dscnn_model import fixture as base_fixture
from test_caption_holdout import SEED


def fixture(root):
    domains,old,new,fitted=previous_fixture(root);base=base_fixture()
    fitted['layers']=make_layers(base,np.ones((7,32),dtype=np.int8))
    return domains,old,new,fitted,base


def setup(old,new,fitted):
    stack=ExitStack()
    stack.enter_context(patch.object(trainer,'load_legacy_arrays',return_value=old))
    stack.enter_context(patch.object(trainer,'load_neural_arrays',return_value=new))
    stack.enter_context(patch.object(trainer,'fit',side_effect=AssertionError('full network fit reached')))
    residual=stack.enter_context(patch.object(trainer,'fit_residual',return_value=fitted.copy()))
    stack.enter_context(patch.object(trainer,'scores',side_effect=fake_scores))
    return stack,residual


class ResidualCandidateTests(unittest.TestCase):
    def test_explicit_frozen_base_routes_cpu_fit_and_records_verified_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted,base=fixture(root);before=frozen_digest(base)
            stack,fit=setup(old,new,fitted)
            with stack:r=trainer.train_candidate(domains,root/'model',holdout_seed=SEED,frontend=MODE,frozen_base=base)
            self.assertIs(fit.call_args.kwargs['base_layers'],base);self.assertTrue(fit.call_args.kwargs['envelope'])
            self.assertTrue(r['base_prefix_unchanged']);self.assertEqual(r['base_numeric_sha256'],before)
            self.assertEqual(r['learning_method'],'frozen_prefix_auxiliary_qat');self.assertEqual(r['numeric_bytes'],23635)
            self.assertTrue(r['accepted_validation']);self.assertIn('arcade_ai_pet/dscnn_residual_training.py',r['source_sha256'])

    def test_modified_returned_prefix_refuses_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted,base=fixture(root);fitted['layers'][0]['weight'].flat[0]+=1
            stack,_=setup(old,new,fitted)
            with stack,self.assertRaisesRegex(ValueError,'frozen convolution changed'):trainer.train_candidate(domains,root/'model',holdout_seed=SEED,frontend=MODE,frozen_base=base)
            self.assertFalse((root/'model').exists())

    def test_mutation_of_original_base_refuses_even_if_the_fitter_did_not_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted,base=fixture(root);stack,fit=setup(old,new,fitted)
            def changed(*args,**kwargs):base[-1]['bias'][0]+=1;return fitted.copy()
            with stack:
                fit.side_effect=changed
                with self.assertRaisesRegex(ValueError,'frozen base changed'):trainer.train_candidate(domains,root/'model',holdout_seed=SEED,frontend=MODE,frozen_base=base)
            self.assertFalse((root/'model').exists())

    def test_base_mutation_during_calibration_refuses_before_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted,base=fixture(root);stack,_=setup(old,new,fitted)
            original=trainer.calibrate
            def changed(*args,**kwargs):
                result=original(*args,**kwargs);base[-1]['bias'][0]+=1;return result
            with stack,patch.object(trainer,'calibrate',side_effect=changed):
                with self.assertRaisesRegex(ValueError,'frozen base changed'):trainer.train_candidate(domains,root/'model',holdout_seed=SEED,frontend=MODE,frozen_base=base)
            self.assertFalse((root/'model').exists())

    def test_legacy_mode_foreign_device_or_wrong_base_refuse_before_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted,base=fixture(root)
            for frontend,device,layers in (('context-logmel64','cpu',base),(MODE,'mps',base),(MODE,'cpu',fitted['layers'])):
                with patch.object(trainer,'load_legacy_arrays',side_effect=AssertionError('PCM reached')):
                    with self.assertRaises(ValueError):trainer.train_candidate(domains,root/'model',holdout_seed=SEED,frontend=frontend,device=device,frozen_base=layers)

    def test_original_neural_regression_still_rejects_the_new_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted,base=fixture(root)
            mask=new['validation']['y']==0;new['validation']['x'][mask,:7]=0;new['validation']['x'][mask,1]=2
            stack,_=setup(old,new,fitted)
            with stack:r=trainer.train_candidate(domains,root/'model',holdout_seed=SEED,frontend=MODE,frozen_base=base)
            self.assertTrue(r['selection_accepted']);self.assertFalse(r['accepted_validation'])

if __name__=='__main__':unittest.main()
