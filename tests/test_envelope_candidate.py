"""新形式を宣言しても、選定/回帰/元データ保護の基準を弱めない。"""
from contextlib import ExitStack
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import dscnn_candidate as trainer
from test_dscnn_candidate import fixture as previous_fixture
from test_envelope_model import fixture as layer_fixture
from test_caption_holdout import SEED

MODE='context-logmel64-envelope'


def fixture(root):
    domains,old,new,fitted=previous_fixture(root)
    for arrays in (old,new):
        for row in arrays.values():row['x']=np.pad(row['x'],((0,0),(0,32)),constant_values=.25)
    fitted['layers']=layer_fixture()
    return domains,old,new,fitted


def fake_scores(values,layers,**kwargs):
    assert kwargs==dict(envelope=True)
    labels=values[:,:7].argmax(1);result=np.zeros((len(values),7),dtype=np.int32);result[np.arange(len(values)),labels]=4096;return result


class EnvelopeCandidateTests(unittest.TestCase):
    def run_candidate(self,root,old,new,fitted,domains):
        with ExitStack() as stack:
            legacy=stack.enter_context(patch.object(trainer,'load_legacy_arrays',return_value=old))
            neural=stack.enter_context(patch.object(trainer,'load_neural_arrays',return_value=new))
            fit=stack.enter_context(patch.object(trainer,'fit',return_value=fitted.copy()))
            stack.enter_context(patch.object(trainer,'scores',side_effect=fake_scores))
            result=trainer.train_candidate(domains,root/'model',holdout_seed=SEED,frontend=MODE)
            return result,legacy,neural,fit

    def test_new_frontend_integer_mode_size_and_metadata_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted=fixture(root)
            r,legacy,neural,fit=self.run_candidate(root,old,new,fitted,domains)
            self.assertEqual(legacy.call_args.kwargs['frontend'],MODE);self.assertEqual(neural.call_args.kwargs['frontend'],MODE)
            self.assertTrue(fit.call_args.kwargs['envelope']);self.assertEqual(r['frontend'],MODE)
            self.assertEqual(r['classifier'],'logmel-dscnn-envelope-v1');self.assertEqual(r['numeric_bytes'],23635)
            self.assertTrue(r['accepted_validation']);self.assertEqual(len(r['domain_profiles']),24)
            self.assertIn('arcade_ai_pet/envelope_frontend.py',r['source_sha256'])
            self.assertFalse(r['final_evaluated']);self.assertFalse(r['device_tested'])
            with np.load(root/'model/weights.npz',allow_pickle=False) as z:self.assertEqual(z['weight5'].shape,(7,3169))

    def test_familiar_regression_failure_still_rejects_the_new_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted=fixture(root)
            mask=new['validation']['y']==0;new['validation']['x'][mask,:7]=0;new['validation']['x'][mask,1]=2
            r,*_=self.run_candidate(root,old,new,fitted,domains)
            self.assertTrue(r['selection_accepted']);self.assertFalse(r['accepted_validation'])

    def test_legacy_export_cannot_be_saved_as_the_new_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted=fixture(root)
            fitted['layers'][-1]['weight']=fitted['layers'][-1]['weight'][:,:3137]
            with self.assertRaises(ValueError):self.run_candidate(root,old,new,fitted,domains)
            self.assertFalse((root/'model').exists())

    def test_foreign_frontend_refuses_before_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,*_=fixture(root)
            with patch.object(trainer,'load_legacy_arrays',side_effect=AssertionError('builder reached')):
                with self.assertRaises(ValueError):trainer.train_candidate(domains,root/'model',holdout_seed=SEED,frontend='foreign')

if __name__=='__main__':unittest.main()
