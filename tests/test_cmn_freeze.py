"""CMNの意味とsourceを固定し、生成・最終評価が同じ経路を使う。"""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
from arcade_ai_pet import cmn_assets, cmn_frontend, decision_assets
from arcade_ai_pet import decision_freeze as freeze, decision_final as final
from test_decision_freeze import fixture, plan
import test_neural_corpus as source_fixture


def patched(root):
    weights, domains, examples, feature, native = fixture(root)
    stack = ExitStack()
    stack.enter_context(patch.object(freeze, 'CmnNativeModel', native))
    stack.enter_context(patch.object(freeze, '_examples', side_effect=lambda path: iter(examples[path])))
    stack.enter_context(patch.object(freeze, '_features', side_effect=feature))
    stack.enter_context(patch.object(freeze, 'simulate', side_effect=lambda pcm,*args: pcm))
    stack.enter_context(patch.object(freeze, 'DecisionNativeModel', side_effect=AssertionError('legacy engine reached')))
    return stack, weights, domains


class CmnFreezeTests(unittest.TestCase):
    def frozen(self, root):
        stack, weights, domains = patched(root)
        with stack:
            result = freeze.freeze_candidate(weights,[10]*5,domains,plan(),root/'frozen',runtime=cmn_assets.RUNTIME)
        return root/'frozen',result

    def test_contract_sources_and_numeric_export_are_bound_to_cmn(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            directory, result = self.frozen(root)
            self.assertEqual(result['runtime'],cmn_assets.RUNTIME)
            self.assertEqual(result['input_contract'],cmn_assets.CONTRACT)
            for name in ('arcade_ai_pet/cmn_frontend.py','arcade_ai_pet/cmn_arrays.py','arcade_ai_pet/cmn_assets.py','arcade_ai_pet/cmn_native.py','arcade_ai_pet/duration_arrays.py','arcade_ai_pet/duration_quantize.py','native/cmn_features.c','native/cmn_features.h'):
                self.assertIn(name,result['source_sha256'])
            self.assertEqual(result['conformance']['rows'],112)
            self.assertTrue(result['conformance']['matched'])
            meta,_ = freeze.verify(directory)
            self.assertEqual(meta,result)
            numeric = json.loads((directory/'decision-metadata.json').read_text())
            self.assertEqual(numeric['runtime'],cmn_assets.RUNTIME)
            self.assertEqual(numeric['input_contract'],cmn_assets.CONTRACT)
            self.assertFalse(result['final_evaluated'])

    def test_real_frontend_dispatch_uses_cmn_after_the_existing_conditioning(self):
        pcm = np.zeros(32000,dtype=np.int16)
        pcm[4000:18000] = (1200*np.sin(np.arange(14000)*.21)).astype(np.int16)
        from arcade_ai_pet.channel_robustness import simulate,condition
        from arcade_ai_pet.channel_training import level
        expected = cmn_frontend.features(level(condition(simulate(pcm,'hum',91),'hp160')))
        np.testing.assert_array_equal(freeze._features(pcm,'hum',91,cmn_assets.RUNTIME),expected)
        legacy = freeze._features(pcm,'hum',91)
        self.assertGreater(float(abs(legacy-expected).max()),.01)
        np.testing.assert_array_equal(legacy[240:],expected[240:])
        with self.assertRaises(ValueError):
            freeze._features(pcm,'hum',91,'untrusted-runtime')

    def test_unknown_runtime_and_failed_cmn_quality_leave_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            weights, domains, _, _, _ = fixture(root)
            with patch.object(freeze,'_examples',side_effect=AssertionError('PCM reached')):
                with self.assertRaises(ValueError):
                    freeze.freeze_candidate(weights,[0]*5,domains,plan(),root/'bad',runtime='unknown')
            self.assertFalse((root/'bad').exists())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            stack, weights, domains = patched(root)
            with stack,patch.object(freeze,'_accepted',return_value=False):
                with self.assertRaises(ValueError):
                    freeze.freeze_candidate(weights,[0]*5,domains,plan(),root/'bad',runtime=cmn_assets.RUNTIME)
            self.assertFalse((root/'bad').exists())

    def test_cmn_source_or_runtime_change_refuses_before_provider(self):
        for mode in ('source','runtime','numeric-contract'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                frozen,_ = self.frozen(root)
                path = frozen/'decision-frozen.json'
                meta = json.loads(path.read_text())
                if mode=='source':
                    meta['source_sha256']['native/cmn_features.c']='0'*64
                elif mode=='runtime':
                    meta['runtime']=decision_assets.RUNTIME
                else:
                    p=frozen/'decision-metadata.json';numeric=json.loads(p.read_text());numeric['input_contract']=decision_assets.CONTRACT
                    p.write_text(json.dumps(numeric));meta['asset_metadata_sha256']=decision_assets.digest(p)
                path.write_text(json.dumps(meta))
                provider = Mock()
                with self.assertRaises(ValueError):
                    final.generate_final(plan(),root/'final',frozen=frozen,synthesize=provider,recognize=provider)
                provider.assert_not_called()
                self.assertFalse((root/'final').exists())

    def test_final_generation_and_evaluation_select_cmn_without_recalibrating(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();frozen,_=self.frozen(root)
            synth,recognize,calls=source_fixture.NeuralCorpusTests().provider()
            output=root/'final'
            final.generate_final(plan(),output,frozen=frozen,synthesize=synth,recognize=recognize)
            final.finalize_final(output,frozen=frozen)
            manifest=json.loads((output/'manifest.json').read_text());data=[]
            for row in manifest['records']:
                data.extend([(np.full(32000,row['label']+1,dtype=np.int16),row['label'],None)]*(manifest['variants'] if 'sha256' in row else 1))
            seen=[]
            def feature(pcm,profile,seed,runtime):
                seen.append(runtime)
                x=np.zeros(261,dtype=np.float32);x[int(pcm[0])-1]=4;return x
            with patch.object(final,'_examples',return_value=iter(data)),patch.object(final,'_features',side_effect=feature),patch.object(final,'_native_audit',return_value=dict(matched=True)),patch('arcade_ai_pet.class_rejection.calibrate',side_effect=AssertionError('final calibration')):
                result=final.evaluate_final(frozen,output,root/'report')
            self.assertEqual(set(seen),{cmn_assets.RUNTIME})
            self.assertTrue(result['accepted_synthetic'])
            self.assertEqual(result['runtime'],cmn_assets.RUNTIME)
            self.assertEqual(result['thresholds_q10'],[10]*5)
            self.assertFalse(result['model_selection_used_final'])
            self.assertFalse(result['device_tested'])

    def test_final_native_audit_routes_through_the_frozen_cmn_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();frozen,_=self.frozen(root)
            with patch.object(final,'_recompute',return_value=([],[],{'matched':True})) as recompute:
                self.assertEqual(final._native_audit(frozen,root/'dataset',[],np.zeros(5,dtype=np.int32)),{'matched':True})
            self.assertEqual(recompute.call_args.kwargs['runtime'],cmn_assets.RUNTIME)


if __name__=='__main__':
    unittest.main()
