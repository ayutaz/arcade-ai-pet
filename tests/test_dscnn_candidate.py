"""2D CNNの実験も固定caption保留/追加回帰/停止条件へ結び付ける。"""
from contextlib import ExitStack
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import dscnn_candidate as trainer
from arcade_ai_pet.decision_assets import read_layers as old_read
from test_caption_training import fixture as previous_fixture
from test_caption_holdout import SEED
from test_dscnn_model import fixture as layers_fixture


def fixture(root):
    domains,old,new,_=previous_fixture(root)
    for arrays in (old,new):
        for row in arrays.values():row['x']=np.pad(row['x'],((0,0),(0,1852)))
    fitted=dict(layers=layers_fixture(),selected_epoch=1,checkpoint_threshold_q10=10,seed=42,training_device='cpu')
    return domains,old,new,fitted


def fake_scores(values,layers):
    labels=values[:,:7].argmax(1);result=np.zeros((len(values),7),dtype=np.int32);result[np.arange(len(values)),labels]=4096;return result


def patches(old,new,fitted):
    stack=ExitStack()
    legacy=stack.enter_context(patch.object(trainer,'load_legacy_arrays',return_value=old))
    neural=stack.enter_context(patch.object(trainer,'load_neural_arrays',return_value=new))
    stack.enter_context(patch.object(trainer,'scores',side_effect=fake_scores))
    fit=stack.enter_context(patch.object(trainer,'fit',return_value=fitted.copy()))
    return stack,fit,legacy,neural


class DscnnCandidateTests(unittest.TestCase):
    def test_fixed_profile_and_held_selection_feed_fit_and_new_numeric_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted=fixture(root);stack,fit,legacy,neural=patches(old,new,fitted)
            with stack:r=trainer.train_candidate(domains,root/'model',holdout_seed=SEED)
            self.assertEqual(neural.call_args.kwargs['frontend'],'context-logmel64')
            np.testing.assert_array_equal(fit.call_args.args[2],np.concatenate((old['validation']['x'],new['held_validation']['x'])))
            self.assertEqual(fit.call_args.kwargs['device'],'cpu');self.assertEqual(fit.call_args.kwargs['max_steps'],8000)
            self.assertTrue(r['accepted_validation']);self.assertEqual(len(r['domain_profiles']),24);self.assertEqual(len(r['pooled_profiles']),8)
            self.assertEqual(r['numeric_bytes'],23411);self.assertEqual(r['classifier'],'logmel-dscnn-v1')
            self.assertFalse(r['final_evaluated']);self.assertFalse(r['device_tested'])
            with self.assertRaises(ValueError):old_read(root/'model/weights.npz')
            with np.load(root/'model/weights.npz',allow_pickle=False) as a:self.assertEqual(len(a.files),18)

    def test_original_neural_regression_still_rejects_an_otherwise_passing_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted=fixture(root)
            new['validation']['x'][new['validation']['y']==0]=0;new['validation']['x'][new['validation']['y']==0,1]=2
            stack,fit,_,_=patches(old,new,fitted)
            with stack:r=trainer.train_candidate(domains,root/'model',holdout_seed=SEED)
            self.assertTrue(r['selection_accepted']);self.assertFalse(r['accepted_validation'])
            self.assertTrue((root/'model/validation.json').exists())

    def test_final_or_existing_output_or_foreign_backend_refuses_before_builders(self):
        for mode in ('final','existing','backend'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();domains,old,new,fitted=fixture(root);kwargs={}
                if mode=='final':p=domains['legacy']/'manifest.json';m=json.loads(p.read_text());m['role']='final';p.write_text(json.dumps(m))
                elif mode=='existing':(root/'model').mkdir()
                else:kwargs['device']='cuda'
                with patch.object(trainer,'load_legacy_arrays',side_effect=AssertionError('builder reached')):
                    with self.assertRaises(ValueError):trainer.train_candidate(domains,root/'model',holdout_seed=SEED,**kwargs)

    def test_extra_test_arrays_and_held_trace_overlap_refuse_before_fit(self):
        for mode in ('test','overlap'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();domains,old,new,fitted=fixture(root)
                if mode=='test':new['test']=new['validation']
                else:new['train']['source_ids'][0]=new['held_validation']['source_ids'][0]
                stack,fit,_,_=patches(old,new,fitted)
                with stack:
                    with self.assertRaises(ValueError):trainer.train_candidate(domains,root/'model',holdout_seed=SEED)
                fit.assert_not_called();self.assertFalse((root/'model').exists())

    def test_row_budget_refuses_before_copying_an_oversized_training_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted=fixture(root)
            old['train']['x']=np.broadcast_to(old['train']['x'][:1],(65536,2113));old['train']['y']=np.zeros(65536,dtype=np.int64)
            stack,fit,_,_=patches(old,new,fitted)
            with stack,patch.object(trainer.np,'tile',side_effect=AssertionError('large allocation reached')):
                with self.assertRaises(ValueError):trainer.train_candidate(domains,root/'model',holdout_seed=SEED)
            fit.assert_not_called()

    def test_fit_timeout_is_not_saved_as_an_accepted_or_partial_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted=fixture(root);stack,fit,_,_=patches(old,new,fitted)
            with stack:
                fit.side_effect=TimeoutError('budget')
                with self.assertRaises(TimeoutError):trainer.train_candidate(domains,root/'model',holdout_seed=SEED)
            self.assertFalse((root/'model').exists())

    def test_input_mutation_after_fit_refuses_candidate_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted=fixture(root);stack,fit,_,_=patches(old,new,fitted)
            def mutate(*args,**kwargs):
                (domains['neural']/'test.npz').write_bytes(b'changed');return fitted.copy()
            with stack:
                fit.side_effect=mutate
                with self.assertRaises(ValueError):trainer.train_candidate(domains,root/'model',holdout_seed=SEED)
            self.assertFalse((root/'model').exists())


if __name__=='__main__':unittest.main()
