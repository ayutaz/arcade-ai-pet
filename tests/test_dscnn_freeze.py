"""先行契約: 保留train条件も含む全24groupをfreezeし、同じ最終生成へ渡す。"""
from contextlib import ExitStack
from pathlib import Path
import hashlib,json,tempfile,unittest
from unittest.mock import Mock,patch
import numpy as np
from arcade_ai_pet import dscnn_freeze as freeze,dscnn_assets as assets
from arcade_ai_pet import decision_freeze as common,decision_final as final
from test_decision_freeze import fixture as old_fixture,plan
from test_dscnn_assets import model
import test_neural_corpus as source_fixture


def patched(root):
    weights,domains,examples,_,_=old_fixture(root);model(weights)
    for directory in domains.values():
        for name in ('train.npz','validation.npz','test.npz'):np.savez(directory/name,fixture_only=np.array([1]))
    path=domains['neural']/'manifest.json';manifest=json.loads(path.read_text());extra=[]
    for label in range(7):
        source=f'held/{label}';row=dict(source=source,id=hashlib.sha256(source.encode()).hexdigest(),label=label,split='train',condition_id='held',sha256=hashlib.sha256(('wave/'+source).encode()).hexdigest());extra.append(row)
        examples[domains['neural']].append((np.full(32000,label+1,np.int16),label,None))
    manifest['records']+=extra;path.write_text(json.dumps(manifest))
    selection=dict(seed=19,manifest_sha256=assets.digest(path),held_condition_ids=['held'],held_control_ids=[])
    def feature(pcm,*args):
        x=np.zeros(2113,np.float32);x[int(pcm[0])-1]=4;return x
    def scores(values,layers):return np.rint(values[:,:7]*1024).astype(np.int32)
    class Native:
        scratch_bytes=26672
        def __init__(self,directory):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def extract(self,pcm,*,banked=False):return feature(pcm)
        def predict(self,values):
            logits=scores(values[None,:],None)[0];return int(logits.argmax()),logits
    stack=ExitStack()
    for name,value in (('_examples',lambda path:iter(examples[path])),('_features',feature),('simulate',lambda pcm,*args:pcm),('scores',scores),('select_holdout',lambda dataset,seed:selection)):
        stack.enter_context(patch.object(freeze,name,side_effect=value))
    stack.enter_context(patch.object(freeze,'NativeModel',Native))
    return stack,weights,domains,examples,selection


class DscnnFreezeTests(unittest.TestCase):
    def frozen(self,root):
        stack,w,domains,_,selection=patched(root)
        with stack:result=freeze.freeze_candidate(w,[10]*5,domains,plan(),root/'frozen',holdout_seed=19)
        return root/'frozen',result

    def test_freeze_recomputes_24_groups_including_held_train_and_exports_exact_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();frozen,r=self.frozen(root)
            self.assertEqual(r['runtime'],assets.RUNTIME);self.assertEqual(r['input_contract'],assets.CONTRACT)
            self.assertEqual(r['numeric_bytes'],23411);self.assertEqual(len(r['domain_profiles']),24)
            self.assertEqual({x['domain'] for x in r['domain_profiles']},{'legacy','held-neural','familiar-neural'})
            self.assertEqual(r['conformance']['rows'],168);self.assertTrue(r['conformance']['matched'])
            self.assertEqual(r['selection']['seed'],19);self.assertFalse(r['final_evaluated'])
            self.assertIn('arcade_ai_pet/dscnn_freeze.py',r['source_sha256'])
            self.assertIn('arcade_ai_pet/host_teacher.py',r['source_sha256'])
            actual,layers=freeze.verify(frozen);self.assertEqual(actual,r);self.assertEqual(len(layers),6)
            self.assertEqual(common.verify(frozen)[0],r)

    def test_bad_held_group_cannot_hide_behind_quality_or_native_success_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();stack,w,domains,examples,_=patched(root)
            examples[domains['neural']][7]=(np.full(32000,2,np.int16),0,None)
            with stack:
                with self.assertRaises(ValueError):freeze.freeze_candidate(w,[0]*5,domains,plan(),root/'bad',holdout_seed=19)
            self.assertFalse((root/'bad').exists())

    def test_missing_seed_or_reserved_condition_overlap_refuses_before_native(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();stack,w,domains,_,_=patched(root)
            with stack,patch.object(freeze,'NativeModel',side_effect=AssertionError('native reached')):
                for seed in (None,True,-1):
                    with self.assertRaises(ValueError):freeze.freeze_candidate(w,[0]*5,domains,plan(),root/'bad',holdout_seed=seed)
                path=domains['neural']/'manifest.json';m=json.loads(path.read_text());m['records'][0]['condition_id']='final-01';path.write_text(json.dumps(m))
                with self.assertRaises(ValueError):freeze.freeze_candidate(w,[0]*5,domains,plan(),root/'bad',holdout_seed=19)
            self.assertFalse((root/'bad').exists())

    def test_modified_source_numeric_headers_plan_or_thresholds_refuse_before_provider(self):
        for kind in ('source','weight','header','plan','threshold'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();directory,r=self.frozen(root);path=directory/'decision-frozen.json'
                if kind=='source':r['source_sha256']['native/logmel_features.c']='0'*64;path.write_text(json.dumps(r))
                elif kind=='weight':(directory/'weights.npz').write_bytes(b'bad')
                elif kind=='header':(directory/'logmel_model.h').write_text('bad')
                elif kind=='plan':(directory/'final-plan.json').write_text('{}')
                else:r['thresholds_q10'][0]=11;path.write_text(json.dumps(r))
                callback=Mock()
                with self.assertRaises(ValueError):final.generate_final(plan(),root/'final',frozen=directory,synthesize=callback,recognize=callback)
                callback.assert_not_called();self.assertFalse((root/'final').exists())

    def test_final_generation_and_finalize_bind_only_the_new_runtime_and_four_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();directory,r=self.frozen(root);synth,recognize,calls=source_fixture.NeuralCorpusTests().provider()
            result=final.generate_final(plan(),root/'final',frozen=directory,synthesize=synth,recognize=recognize)
            self.assertTrue(result['final_conditions_generated']);self.assertEqual(len({c[1] for c in calls}),4)
            final.finalize_final(root/'final',frozen=directory)
            m=json.loads((root/'final/manifest.json').read_text());self.assertEqual(m['model_binding'],final.binding(directory));self.assertEqual(m['role'],'final')

    def test_common_runtime_dispatch_uses_logmel_features_cnn_scores_and_native_final_audit(self):
        pcm=np.zeros(32000,np.int16)
        self.assertEqual(common._profile(assets.RUNTIME)[0],assets)
        self.assertEqual(common._features(pcm,'clean',1,assets.RUNTIME).shape,(2113,))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();layers=model(root/'weights.npz')
            self.assertEqual(common._scores(np.zeros((1,2113),np.float32),layers,assets.RUNTIME).shape,(1,7))
            directory,_=self.frozen(root)
            with patch.object(final,'_recompute',return_value=([],[],{'matched':True})) as recompute:
                self.assertTrue(final._native_audit(directory,root/'data',layers,np.zeros(5,np.int32))['matched'])
            self.assertEqual(recompute.call_args.kwargs['runtime'],assets.RUNTIME)
        with self.assertRaises(ValueError):common._scores(np.zeros((1,2113),np.float32),layers,'unknown')

    def test_final_evaluation_uses_cnn_dispatch_without_calibration_or_threshold_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();directory,_=self.frozen(root);synth,recognize,_=source_fixture.NeuralCorpusTests().provider()
            dataset=root/'final';final.generate_final(plan(),dataset,frozen=directory,synthesize=synth,recognize=recognize);final.finalize_final(dataset,frozen=directory)
            manifest=json.loads((dataset/'manifest.json').read_text());data=[]
            for row in manifest['records']:data.extend([(np.full(32000,row['label']+1,np.int16),row['label'],None)]*(manifest['variants'] if 'sha256' in row else 1))
            def feature(pcm,*args):
                x=np.zeros(2113,np.float32);x[int(pcm[0])-1]=4;return x
            seen=[]
            def cnn(values,layers,runtime):
                seen.append(runtime);return np.rint(values[:,:7]*1024).astype(np.int32)
            with patch.object(final,'_examples',return_value=iter(data)),patch.object(final,'_features',side_effect=feature),patch.object(final,'_scores',side_effect=cnn),patch.object(final,'_native_audit',return_value={'matched':True}),patch('arcade_ai_pet.class_rejection.calibrate',side_effect=AssertionError('final calibration')):
                result=final.evaluate_final(directory,dataset,root/'report')
            self.assertEqual(set(seen),{assets.RUNTIME});self.assertTrue(result['accepted_synthetic'])
            self.assertEqual(result['thresholds_q10'],[10]*5);self.assertFalse(result['model_selection_used_final'])

if __name__=='__main__':unittest.main()
