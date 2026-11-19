"""先行契約: 新headも既存の保留/品質/回帰/停止条件へ結び付ける。"""
from contextlib import ExitStack
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import dscnn_candidate as trainer
from test_dscnn_candidate import fixture as previous_fixture
from test_time_pool_model import fixture as layer_fixture
from test_caption_holdout import SEED

MODE=dict(envelope=True,time_pool=True)
FRONTEND='context-logmel64-envelope'


def fixture(root):
    domains,old,new,fitted=previous_fixture(root)
    for arrays in (old,new):
        for row in arrays.values():row['x']=np.pad(row['x'],((0,0),(0,32)),constant_values=.25)
    fitted['layers']=layer_fixture()
    return domains,old,new,fitted


def fake_scores(values,layers,**kwargs):
    assert kwargs==MODE
    labels=values[:,:7].argmax(1);out=np.zeros((len(values),7),dtype=np.int32);out[np.arange(len(values)),labels]=4096;return out


def patches(old,new,fitted):
    stack=ExitStack()
    legacy=stack.enter_context(patch.object(trainer,'load_legacy_arrays',return_value=old))
    neural=stack.enter_context(patch.object(trainer,'load_neural_arrays',return_value=new))
    fit=stack.enter_context(patch.object(trainer,'fit',return_value=fitted.copy()))
    stack.enter_context(patch.object(trainer,'scores',side_effect=fake_scores))
    return stack,legacy,neural,fit


class TimePoolCandidateTests(unittest.TestCase):
    def test_same_features_new_classifier_capacity_and_integer_selection_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted=fixture(root);stack,legacy,neural,fit=patches(old,new,fitted)
            with stack:r=trainer.train_candidate(domains,root/'model',holdout_seed=SEED,frontend=FRONTEND,time_pool=True)
            self.assertEqual(legacy.call_args.kwargs['frontend'],FRONTEND);self.assertEqual(neural.call_args.kwargs['frontend'],FRONTEND)
            self.assertTrue(fit.call_args.kwargs['time_pool']);self.assertTrue(fit.call_args.kwargs['envelope'])
            self.assertEqual(r['classifier'],'logmel-dscnn-timepool-v1');self.assertEqual(r['numeric_bytes'],7507)
            self.assertEqual(r['time_pool_bins'],4);self.assertEqual(r['time_pool_integer_divisor'],4)
            self.assertTrue(r['accepted_validation']);self.assertEqual(len(r['domain_profiles']),24)
            self.assertFalse(r['final_evaluated']);self.assertFalse(r['device_tested'])
            with np.load(root/'model/weights.npz',allow_pickle=False) as z:self.assertEqual(z['weight5'].shape,(7,865))

    def test_original_neural_regression_still_rejects_otherwise_passing_new_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted=fixture(root)
            mask=new['validation']['y']==0;new['validation']['x'][mask,:7]=0;new['validation']['x'][mask,1]=2
            stack,*_=patches(old,new,fitted)
            with stack:r=trainer.train_candidate(domains,root/'model',holdout_seed=SEED,frontend=FRONTEND,time_pool=True)
            self.assertTrue(r['selection_accepted']);self.assertFalse(r['accepted_validation'])

    def test_old_head_cannot_be_saved_as_the_new_classifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted=fixture(root)
            fitted['layers'][-1]['weight']=np.zeros((7,3169),dtype=np.int8)
            stack,*_=patches(old,new,fitted)
            with stack:
                with self.assertRaises(ValueError):trainer.train_candidate(domains,root/'model',holdout_seed=SEED,frontend=FRONTEND,time_pool=True)
            self.assertFalse((root/'model').exists())

    def test_unsupported_mode_or_frozen_prefix_rejects_before_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,*_=fixture(root)
            for options in ({'time_pool':True},{'frontend':FRONTEND,'time_pool':1},{'frontend':FRONTEND,'time_pool':True,'frozen_base':layer_fixture()}):
                with patch.object(trainer,'load_legacy_arrays',side_effect=AssertionError('builder reached')):
                    with self.assertRaises(ValueError):trainer.train_candidate(domains,root/'model',holdout_seed=SEED,**options)

    def test_held_trace_overlap_and_fit_timeout_never_save_partial_output(self):
        for mode in ('overlap','timeout'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();domains,old,new,fitted=fixture(root)
                if mode=='overlap':new['train']['source_ids'][0]=new['held_validation']['source_ids'][0]
                stack,_,_,fit=patches(old,new,fitted)
                with stack:
                    if mode=='timeout':fit.side_effect=TimeoutError('budget')
                    with self.assertRaises(ValueError if mode=='overlap' else TimeoutError):trainer.train_candidate(domains,root/'model',holdout_seed=SEED,frontend=FRONTEND,time_pool=True)
                if mode=='overlap':fit.assert_not_called()
                self.assertFalse((root/'model').exists())

if __name__=='__main__':unittest.main()
