"""候補選定に保留captionを使い、元neural検証は別の必須回帰にする。"""
from contextlib import ExitStack
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import caption_training as trainer
from arcade_ai_pet.channel_robustness import PROFILES
from arcade_ai_pet.decision_assets import read_layers
from test_caption_holdout import fixture as neural_fixture,SEED


def values(value,validation=False,identity='train'):
    y=np.tile(np.arange(7),8 if validation else 1).astype(np.int64)
    x=np.zeros((len(y),261),dtype=np.float32);x[np.arange(len(y)),y]=value
    return dict(x=x,y=y,profiles=[p for p in PROFILES for _ in range(7)] if validation else ['clean']*7,
                source_ids=[hashlib.sha256(f'{identity}/{i}'.encode()).hexdigest() for i in range(len(y))])


def fixture(root):
    domains={name:root/name for name in ('legacy','neural')}
    for path in domains.values():path.mkdir()
    legacy=dict(role='development',records=[dict(id=hashlib.sha256(f'legacy/{i}'.encode()).hexdigest(),source=f'legacy/{i}',label=i,split='train') for i in range(7)])
    (domains['legacy']/'manifest.json').write_text(json.dumps(legacy))
    for split in ('train','validation','test'):(domains['legacy']/(split+'.npz')).write_bytes(split.encode())
    neural_fixture(domains['neural'])
    old=dict(train=values(4),validation=values(4,True,'legacy-val'))
    new=dict(train=values(3.5,identity='neural-train'),validation=values(2,True,'familiar'),held_validation=values(3,True,'held'))
    layers=[dict(weight=np.pad(np.eye(7,dtype=np.int8),((0,0),(0,254))),bias=np.zeros(7,dtype=np.int32),shift=0),dict(weight=np.eye(7,dtype=np.int8),bias=np.zeros(7,dtype=np.int32),shift=0)]
    fitted=dict(layers=layers,selected_epoch=1,checkpoint_threshold_q10=10,seed=42)
    return domains,old,new,fitted


def patches(old,new,fitted):
    stack=ExitStack();stack.enter_context(patch.object(trainer,'load_legacy_arrays',return_value=old));stack.enter_context(patch.object(trainer,'load_neural_arrays',return_value=new))
    fit=stack.enter_context(patch.object(trainer,'fit',return_value=fitted.copy()))
    return stack,fit


class CaptionTrainingTests(unittest.TestCase):
    def test_fit_and_calibration_use_legacy_and_held_validation_but_not_familiar_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted=fixture(root);stack,fit=patches(old,new,fitted)
            with stack:result=trainer.train_candidate(domains,root/'model',holdout_seed=SEED)
            x,y,vx,vy,groups=fit.call_args.args
            np.testing.assert_array_equal(x,np.concatenate((old['train']['x'],np.tile(new['train']['x'],(2,1)))))
            np.testing.assert_array_equal(vx,np.concatenate((old['validation']['x'],new['held_validation']['x'])))
            np.testing.assert_array_equal(groups,np.repeat(np.arange(16),7))
            self.assertNotIn(2,np.max(vx,axis=1));self.assertEqual(fit.call_args.kwargs['feature_count'],261)
            self.assertTrue(result['accepted_validation']);self.assertEqual(len(result['domain_profiles']),24);self.assertEqual(len(result['pooled_profiles']),8)
            self.assertEqual(result['selection_domains'],['legacy_validation','held_neural_validation'])
            self.assertEqual(result['additional_required_regression'],['original_neural_validation'])
            self.assertFalse(result['final_evaluated']);self.assertFalse(result['device_tested'])
            self.assertTrue(result['original_archives_unchanged']);self.assertTrue((root/'model/validation.json').is_file())
            self.assertEqual(read_layers(root/'model/weights.npz')[0][0]['weight'].shape,(7,261))

    def test_familiar_regression_failure_saves_negative_result_without_affecting_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted=fixture(root)
            new['validation']['x'][new['validation']['y']==0]=0;new['validation']['x'][new['validation']['y']==0,1]=2
            stack,fit=patches(old,new,fitted)
            with stack:result=trainer.train_candidate(domains,root/'model',holdout_seed=SEED)
            self.assertTrue(result['selection_accepted']);self.assertFalse(result['accepted_validation'])
            self.assertEqual(result['selected_epoch'],1);self.assertTrue((root/'model/validation.json').is_file())
            np.testing.assert_array_equal(fit.call_args.args[2],np.concatenate((old['validation']['x'],new['held_validation']['x'])))

    def test_final_role_or_cross_domain_identity_overlap_refuses_before_builders(self):
        for mode in ('legacy-final','neural-final','wave-overlap','id-overlap'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();domains,old,new,fitted=fixture(root)
                a=json.loads((domains['legacy']/'manifest.json').read_text());b=json.loads((domains['neural']/'manifest.json').read_text())
                if mode=='legacy-final':a['role']='final'
                elif mode=='neural-final':b['role']='final'
                elif mode=='id-overlap':a['records'][0]['id']=b['records'][0]['id']
                else:a['records'][0]['sha256']=b['records'][0]['sha256']
                (domains['legacy']/'manifest.json').write_text(json.dumps(a));(domains['neural']/'manifest.json').write_text(json.dumps(b))
                with patch.object(trainer,'load_legacy_arrays',side_effect=AssertionError('PCM reached')):
                    with self.assertRaises(ValueError):trainer.train_candidate(domains,root/'model',holdout_seed=SEED)
                self.assertFalse((root/'model').exists())

    def test_existing_output_and_invalid_recipe_are_refused_before_training(self):
        for options in ({'epochs':0},{'hidden':91},{'neural_weight':True},{'holdout_seed':True},{'max_passes':9},{'existing':True}):
            with self.subTest(options=options),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();domains,old,new,fitted=fixture(root);args=dict(holdout_seed=SEED)
                if options.get('existing'):(root/'model').mkdir()
                else:args.update(options)
                with patch.object(trainer,'load_legacy_arrays',side_effect=AssertionError('builder reached')):
                    with self.assertRaises(ValueError):trainer.train_candidate(domains,root/'model',**args)

    def test_test_arrays_are_never_forwarded_to_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted=fixture(root);new['test']=values(1,True)
            stack,fit=patches(old,new,fitted)
            with stack:
                with self.assertRaises(ValueError):trainer.train_candidate(domains,root/'model',holdout_seed=SEED)
            fit.assert_not_called();self.assertFalse((root/'model').exists())

    def test_trace_overlap_is_refused_even_if_a_builder_supplies_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted=fixture(root);new['train']['source_ids'][0]=new['held_validation']['source_ids'][0]
            stack,fit=patches(old,new,fitted)
            with stack:
                with self.assertRaises(ValueError):trainer.train_candidate(domains,root/'model',holdout_seed=SEED)
            fit.assert_not_called();self.assertFalse((root/'model').exists())

    def test_changed_input_after_fit_does_not_publish_a_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,old,new,fitted=fixture(root);stack,fit=patches(old,new,fitted)
            def mutate(*args,**kwargs):
                (domains['neural']/'validation.npz').write_bytes(b'changed during fit')
                return fitted.copy()
            with stack:
                fit.side_effect=mutate
                with self.assertRaises(ValueError):trainer.train_candidate(domains,root/'model',holdout_seed=SEED)
            self.assertFalse((root/'model').exists())


if __name__=='__main__':unittest.main()
