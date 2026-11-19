"""開発用の生成条件・内容点検・元発話分離。音声品質は模擬しない。"""
from pathlib import Path
import copy,hashlib,json,tempfile,unittest,wave
from unittest.mock import Mock
import numpy as np
from arcade_ai_pet.neural_corpus import generate_development
import arcade_ai_pet.neural_corpus as corpus
from arcade_ai_pet.data import SOURCES
from arcade_ai_pet.features import split_for
from arcade_ai_pet.training import load_dataset
from arcade_ai_pet.native_evaluation import _examples


def plan():
    return dict(schema=1,seed=20261119,variants=1,attempts=2,
                provider=dict(name='irodori-v2-voice-design',license='MIT',engine_commit='1'*40,
                              model_sha256='2'*64,codec_sha256='3'*64,runtime_lock_sha256='4'*64,
                              asr_model_sha256='5'*64),
                conditions=[dict(id='dev-01',role='development',caption='低めで落ち着いた声。'),
                            dict(id='dev-02',role='development',caption='明るく高めの声。'),
                            dict(id='final-01',role='final',caption='柔らかくゆっくりした声。')])


class NeuralCorpusTests(unittest.TestCase):
    def provider(self):
        lookup={};calls=[]
        def synthesize(prompt,caption,seed,path):
            calls.append((prompt,caption,seed))
            values=np.rint(2000*np.sin(np.arange(3200)*(0.03+(seed%10000)/100000))).astype('<i2')
            lookup[hashlib.sha256(values.tobytes()).hexdigest()]=prompt
            with wave.open(str(path),'wb') as stream:
                stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(16000);stream.writeframes(values.tobytes())
        def recognize(rate,values):return lookup[hashlib.sha256(values.astype('<i2').tobytes()).hexdigest()]
        return synthesize,recognize,calls

    def test_only_development_conditions_generate_and_legacy_reconstruction_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp).resolve()/'dataset';synthesize,recognize,calls=self.provider()
            result=generate_development(plan(),output,synthesize=synthesize,recognize=recognize)
            self.assertEqual(len(calls),42)
            self.assertNotIn('柔らかくゆっくりした声。',{r[1] for r in calls})
            manifest=json.loads((output/'manifest.json').read_text());self.assertEqual(manifest['role'],'development')
            self.assertEqual(manifest['variants'],1);self.assertEqual(result['qualified_speech_sources'],42)
            self.assertFalse(result['real_voice_tested']);self.assertFalse(result['model_trained'])
            for row in manifest['records']:
                self.assertEqual(row['split'],split_for(row['source']))
                self.assertEqual(row['id'],hashlib.sha256(row['source'].encode()).hexdigest())
            data=load_dataset(output);self.assertEqual(sum(len(r['y']) for r in data.values()),242)
            self.assertEqual(len(list(_examples(output))),242)

    def test_bad_plan_existing_output_or_shared_caption_never_calls_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();callback=Mock()
            bad=plan();bad['conditions'][2]['caption']=bad['conditions'][0]['caption']
            for value in (bad,{**plan(),'attempts':True},{**plan(),'provider':{**plan()['provider'],'license':'unreviewed'}}):
                with self.assertRaises(ValueError):generate_development(value,root/'new',synthesize=callback,recognize=callback)
            with self.assertRaises(ValueError):generate_development(plan(),root,synthesize=callback,recognize=callback)
            callback.assert_not_called();self.assertFalse((root/'new').exists())

    def test_mismatch_is_retained_and_retry_never_relabels_or_leaks_final_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp).resolve()/'dataset';synthesize,recognize,calls=self.provider();first=[True]
            def audit(rate,values):
                if first[0]:first[0]=False;return 'まって'
                return recognize(rate,values)
            generate_development(plan(),output,synthesize=synthesize,recognize=audit)
            self.assertEqual(len(calls),43)
            progress=json.loads((output/'progress.json').read_text())
            rejected=[r for r in progress['attempts'] if not r['content_check_passed']]
            self.assertEqual(len(rejected),1);self.assertEqual(rejected[0]['label'],0)
            manifest=json.loads((output/'manifest.json').read_text())
            self.assertNotIn(rejected[0]['id'],{r['id'] for r in manifest['records']})
            self.assertTrue((output/'sources'/(rejected[0]['id']+'.wav')).exists())

    def test_provider_exception_stops_once_and_saves_safe_incomplete_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp).resolve()/'dataset';callback=Mock(side_effect=RuntimeError('private details'))
            with self.assertRaises(RuntimeError):generate_development(plan(),output,synthesize=callback,recognize=Mock())
            callback.assert_called_once();self.assertFalse((output/'manifest.json').exists())
            report=json.loads((output/'progress.json').read_text())
            self.assertEqual(report['status'],'incomplete');self.assertNotIn('private',json.dumps(report))

    def test_no_complete_corpus_when_every_recognition_disagrees(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp).resolve()/'dataset';synthesize,_,_=self.provider()
            with self.assertRaises(ValueError):generate_development(plan(),output,synthesize=synthesize,recognize=lambda *args:'不一致')
            self.assertFalse((output/'manifest.json').exists())
            self.assertEqual(json.loads((output/'progress.json').read_text())['status'],'incomplete')

    def test_qualification_and_feature_finalization_are_separate_without_resynthesis(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp).resolve()/'dataset';synthesize,recognize,calls=self.provider()
            generate_development(plan(),output,synthesize=synthesize,recognize=recognize,finalize=False)
            self.assertTrue((output/'qualified-sources.json').exists())
            self.assertFalse((output/'manifest.json').exists());self.assertFalse((output/'train.npz').exists())
            corpus.finalize_development(output)
            self.assertEqual(len(calls),42);self.assertEqual(len(list(_examples(output))),242)
            self.assertEqual(json.loads((output/'manifest.json').read_text())['feature_numpy_version'],np.__version__)

    def test_changed_qualified_wave_cannot_be_finalized_or_resynthesized(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp).resolve()/'dataset';synthesize,recognize,calls=self.provider()
            generate_development(plan(),output,synthesize=synthesize,recognize=recognize,finalize=False)
            qualified=json.loads((output/'qualified-sources.json').read_text())
            path=output/'sources'/(qualified['records'][0]['id']+'.wav');path.write_bytes(b'changed')
            with self.assertRaises(ValueError):corpus.finalize_development(output)
            self.assertFalse((output/'manifest.json').exists());self.assertFalse((output/'train.npz').exists())
            self.assertEqual(len(calls),42)

    def test_qualification_requires_matching_passed_audit_before_features(self):
        for change in ('missing','failed','different-hash'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                output=Path(tmp).resolve()/'dataset';synthesize,recognize,_=self.provider()
                generate_development(plan(),output,synthesize=synthesize,recognize=recognize,finalize=False)
                path=output/'progress.json';progress=json.loads(path.read_text())
                if change=='missing':progress['attempts']=[]
                elif change=='failed':progress['attempts'][0]['content_check_passed']=False
                else:progress['attempts'][0]['wave_sha256']='0'*64
                path.write_text(json.dumps(progress))
                with self.assertRaises(ValueError):corpus.finalize_development(output)
                self.assertFalse((output/'manifest.json').exists());self.assertFalse((output/'train.npz').exists())


if __name__=='__main__':unittest.main()
