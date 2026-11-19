"""最終生成/特徴化/評価を同一の新形式freezeへ結び付ける。"""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
import numpy as np
from arcade_ai_pet import decision_final as final,decision_freeze as freeze
from test_decision_freeze import plan,patched
import test_neural_corpus as source_fixture


class DecisionFinalTests(unittest.TestCase):
    def frozen(self,root):
        stack,w,domains,_=patched(root)
        with stack:freeze.freeze_candidate(w,[10]*5,domains,plan(),root/'frozen')
        return root/'frozen'

    def test_only_four_frozen_conditions_generate_and_finalization_preserves_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();frozen=self.frozen(root);output=root/'final'
            synth,recognize,calls=source_fixture.NeuralCorpusTests().provider()
            final.generate_final(plan(),output,frozen=frozen,synthesize=synth,recognize=recognize)
            self.assertGreaterEqual(len(calls),84);self.assertLessEqual(len(calls),168)
            self.assertEqual(len({c[1] for c in calls}),4);before=len(calls)
            self.assertFalse((output/'manifest.json').exists())
            q=json.loads((output/'qualified-sources.json').read_text());self.assertEqual(q['model_binding'],final.binding(frozen))
            final.finalize_final(output,frozen=frozen)
            m=json.loads((output/'manifest.json').read_text());self.assertEqual(m['role'],'final');self.assertEqual(m['model_binding'],q['model_binding'])
            self.assertEqual(len(calls),before)

    def test_changed_plan_or_weights_refuse_before_any_provider_or_output(self):
        for mode in ('plan','weights'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();frozen=self.frozen(root);callback=Mock();p=plan()
                if mode=='plan':p['conditions'][-1]['caption']='変更された条件。'
                else:(frozen/'weights.npz').write_bytes(b'changed')
                with self.assertRaises(ValueError):final.generate_final(p,root/'final',frozen=frozen,synthesize=callback,recognize=callback)
                callback.assert_not_called();self.assertFalse((root/'final').exists())

    def test_metadata_changed_after_generation_refuses_feature_finalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();frozen=self.frozen(root);output=root/'final';synth,rec,_=source_fixture.NeuralCorpusTests().provider()
            final.generate_final(plan(),output,frozen=frozen,synthesize=synth,recognize=rec)
            path=frozen/'decision-frozen.json';m=json.loads(path.read_text());m['development_manifest_sha256']['legacy']='0'*64;path.write_text(json.dumps(m))
            with self.assertRaises(ValueError):final.finalize_final(output,frozen=frozen)
            self.assertFalse((output/'manifest.json').exists())

    def dataset(self,root,frozen):
        output=root/'final';synth,recognize,_=source_fixture.NeuralCorpusTests().provider()
        final.generate_final(plan(),output,frozen=frozen,synthesize=synth,recognize=recognize)
        final.finalize_final(output,frozen=frozen)
        return output

    def test_relabelled_or_duplicate_final_metadata_refuses_before_pcm(self):
        for mode in ('duplicate','source','condition','provider'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();frozen=self.frozen(root);dataset=self.dataset(root,frozen)
                path=dataset/'manifest.json';m=json.loads(path.read_text())
                if mode=='duplicate':m['records'][1]=m['records'][0].copy()
                elif mode=='source':m['records'][0]['source']='substituted/source'
                elif mode=='condition':m['records'][0]['condition_id']='final-02'
                else:m['provider']['model_sha256']='0'*64
                path.write_text(json.dumps(m))
                with patch.object(final,'_examples',side_effect=AssertionError('invalid metadata reached PCM')):
                    with self.assertRaises(ValueError):final.evaluate_final(frozen,dataset,root/'report')
                self.assertFalse((root/'report').exists())

    def test_evaluation_checks_binding_and_source_or_wave_overlap_before_pcm(self):
        for mode in ('binding','source','wave','role'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();frozen=self.frozen(root);meta,_=freeze.verify(frozen)
                dataset=root/'final';dataset.mkdir();record=dict(id=hashlib.sha256(b'final/0').hexdigest(),sha256='d'*64,label=0,condition_id='final-01',split='test')
                m=dict(role='final',variants=1,records=[record],plan_sha256=meta['plan_sha256'],model_binding=final.binding(frozen))
                if mode=='binding':m['model_binding']={}
                elif mode=='source':record['id']=meta['source_ids'][0]
                elif mode=='wave':record['sha256']=meta['wave_hashes'][0]
                else:m['role']='development'
                (dataset/'manifest.json').write_text(json.dumps(m))
                with patch.object(final,'_examples',side_effect=AssertionError('PCM reached')):
                    with self.assertRaises(ValueError):final.evaluate_final(frozen,dataset,root/'report')
                self.assertFalse((root/'report').exists())

    def test_final_evaluation_never_calibrates_and_records_failed_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();frozen=self.frozen(root);dataset=self.dataset(root,frozen)
            manifest=json.loads((dataset/'manifest.json').read_text());data=[]
            for row in manifest['records']:
                data.extend([(np.full(32000,row['label']+1,dtype=np.int16),row['label'],None)]*(manifest['variants'] if 'sha256' in row else 1))
            # Independent final data with one deliberately wrong class in every row.
            def wrong(pcm,*args):
                x=np.zeros(261,dtype=np.float32);x[(int(pcm[0]))%7]=4;return x
            with patch.object(final,'_examples',return_value=iter(data)),patch.object(final,'_features',side_effect=wrong),patch.object(final,'_native_audit',return_value=dict(rows=len(data)*8,matched=True,maximum_feature_error=0)),patch('arcade_ai_pet.class_rejection.calibrate',side_effect=AssertionError('no final calibration')):
                result=final.evaluate_final(frozen,dataset,root/'report')
            self.assertFalse(result['accepted_synthetic']);self.assertEqual(len(result['profiles']),8)
            self.assertEqual(result['thresholds_q10'],[10]*5);self.assertFalse(result['device_tested']);self.assertTrue(result['final_evaluated'])
            self.assertTrue((root/'report/decision-final-evaluation.json').exists())

    def test_manifest_mutation_during_final_evaluation_leaves_no_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();frozen=self.frozen(root);dataset=self.dataset(root,frozen)
            path=dataset/'manifest.json';m=json.loads(path.read_text());data=[]
            for row in m['records']:
                data.extend([(np.full(32000,row['label']+1,dtype=np.int16),row['label'],None)]*(m['variants'] if 'sha256' in row else 1))
            def feature(pcm,*args):
                x=np.zeros(261,dtype=np.float32);x[int(pcm[0])-1]=4;return x
            def mutate(*args):
                changed=json.loads(path.read_text());changed['unexpected']=True;path.write_text(json.dumps(changed))
                return dict(rows=len(data)*8,matched=True,maximum_feature_error=0)
            with patch.object(final,'_examples',return_value=iter(data)),patch.object(final,'_features',side_effect=feature),patch.object(final,'_native_audit',side_effect=mutate):
                with self.assertRaises(ValueError):final.evaluate_final(frozen,dataset,root/'report')
            self.assertFalse((root/'report').exists())


if __name__=='__main__':unittest.main()
