"""先行契約: 学習だけのmaskを明示し、未宣言/虚偽recipeを保存しない。"""
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import dscnn_candidate as trainer
from test_dscnn_candidate import fixture,patches
from test_caption_holdout import SEED


RECIPE=dict(time_max=6,frequency_max=3,fill=0,seed=43,training_only=True)


class SpectralMaskCandidateTests(unittest.TestCase):
    def test_opt_in_passes_only_training_flag_keeps_validation_and_records_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted=fixture(root);fitted['spectral_mask']=RECIPE.copy()
            stack,fit,_,_=patches(old,new,fitted)
            with stack:r=trainer.train_candidate(domains,root/'model',holdout_seed=SEED,spectral_mask=True)
            self.assertTrue(fit.call_args.kwargs['spectral_mask']);self.assertNotIn('spectral_mask',fit.call_args.kwargs.get('mode',{}))
            np.testing.assert_array_equal(fit.call_args.args[2],np.concatenate((old['validation']['x'],new['held_validation']['x'])))
            self.assertEqual(r['spectral_mask'],RECIPE);self.assertIn('arcade_ai_pet/spectral_mask.py',r['source_sha256'])
            self.assertEqual(r['classifier'],'logmel-dscnn-v1');self.assertEqual(r['numeric_bytes'],23411)
            self.assertEqual(len(r['domain_profiles']),24);self.assertTrue(r['accepted_validation'])
            self.assertFalse(r['final_evaluated']);self.assertFalse(r['device_tested'])

    def test_default_has_no_flag_or_recipe_and_undeclared_recipe_refuses(self):
        for declared in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();domains,old,new,fitted=fixture(root)
                if declared:fitted['spectral_mask']=RECIPE.copy()
                stack,fit,_,_=patches(old,new,fitted)
                with stack:
                    if declared:
                        with self.assertRaises(ValueError):trainer.train_candidate(domains,root/'model',holdout_seed=SEED)
                    else:
                        r=trainer.train_candidate(domains,root/'model',holdout_seed=SEED)
                        self.assertNotIn('spectral_mask',r)
                self.assertNotIn('spectral_mask',fit.call_args.kwargs)
                if declared:self.assertFalse((root/'model').exists())

    def test_false_recipe_or_calibration_mutation_refuses_before_output(self):
        for kind in ('missing','wrong_seed','evaluation_enabled','mutation'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();domains,old,new,fitted=fixture(root)
                if kind!='missing':fitted['spectral_mask']=RECIPE.copy()
                if kind=='wrong_seed':fitted['spectral_mask']['seed']=42
                if kind=='evaluation_enabled':fitted['spectral_mask']['training_only']=False
                stack,*_=patches(old,new,fitted);original=trainer.calibrate
                def changed(*args,**kwargs):
                    out=original(*args,**kwargs);fitted['spectral_mask']['frequency_max']=4;return out
                with stack,patch.object(trainer,'calibrate',side_effect=changed if kind=='mutation' else original):
                    with self.assertRaises(ValueError):trainer.train_candidate(domains,root/'model',holdout_seed=SEED,spectral_mask=True)
                self.assertFalse((root/'model').exists())

    def test_bad_flags_or_other_modes_refuse_before_builders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,*_=fixture(root)
            for options in ({'spectral_mask':1},{'spectral_mask':'yes'},{'spectral_mask':True,'frontend':'context-logmel64-envelope'},{'spectral_mask':True,'frozen_base':[]}):
                with patch.object(trainer,'load_legacy_arrays',side_effect=AssertionError('builder reached')):
                    with self.assertRaises(ValueError):trainer.train_candidate(domains,root/'model',holdout_seed=SEED,**options)

if __name__=='__main__':unittest.main()
