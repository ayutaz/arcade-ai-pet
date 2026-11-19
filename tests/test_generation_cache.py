"""中断済み生成のWAVだけをhashで再利用し、ASR点検は再実行する。"""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock
from arcade_ai_pet import generation_cache as cache
from arcade_ai_pet.neural_corpus import generate_development
import test_neural_corpus as fixture


class GenerationCacheTests(unittest.TestCase):
    def fixture(self,root):
        plan=fixture.plan();synthesize,recognize,_=fixture.NeuralCorpusTests().provider();prior=root/'prior'
        generate_development(plan,prior,synthesize=synthesize,recognize=recognize,finalize=False)
        progress=json.loads((prior/'progress.json').read_text());all_rows=progress['attempts'];progress.update(status='incomplete',attempts=all_rows[:2])
        (prior/'progress.json').write_text(json.dumps(progress));(prior/'qualified-sources.json').unlink()
        output=root/'new';output.mkdir()
        return plan,prior,output,synthesize,all_rows

    def test_reuses_only_matching_recorded_wave_and_missing_entry_calls_teacher(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();plan,prior,out,provider,rows=self.fixture(root);teacher=Mock(side_effect=provider)
            synthesize,stats=cache.source_callback(plan,prior,synthesize=teacher)
            for row in (rows[0],rows[3]):
                caption=next(c['caption'] for c in plan['conditions'] if c['id']==row['condition_id'])
                synthesize(row['prompt'],caption,row['seed'],out/(row['id']+'.wav'))
            self.assertEqual(stats,dict(reused=1,synthesized=1));teacher.assert_called_once()
            self.assertEqual((out/(rows[0]['id']+'.wav')).read_bytes(),(prior/'sources'/(rows[0]['id']+'.wav')).read_bytes())
            self.assertFalse((out/'qualified-sources.json').exists())
            self.assertEqual(json.loads((prior/'progress.json').read_text())['status'],'incomplete')

    def test_wrong_plan_or_final_request_never_reuses_or_generates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();plan,prior,out,_,rows=self.fixture(root);teacher=Mock();changed=copy.deepcopy(plan);changed['seed']+=1
            with self.assertRaises(ValueError):cache.source_callback(changed,prior,synthesize=teacher)
            synthesize,_=cache.source_callback(plan,prior,synthesize=teacher)
            row=rows[0];final=plan['conditions'][-1]
            with self.assertRaises(ValueError):synthesize(row['prompt'],final['caption'],row['seed'],out/(row['id']+'.wav'))
            teacher.assert_not_called()

    def test_changed_cached_wave_and_existing_output_never_trigger_fallback(self):
        for mode in ('changed','existing','wrong-name'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();plan,prior,out,_,rows=self.fixture(root);teacher=Mock();synthesize,_=cache.source_callback(plan,prior,synthesize=teacher)
                row=rows[0];caption=plan['conditions'][0]['caption'];path=out/(row['id']+'.wav')
                if mode=='changed':(prior/'sources'/(row['id']+'.wav')).write_bytes(b'changed')
                elif mode=='existing':path.write_bytes(b'preserve')
                else:path=out/('0'*64+'.wav')
                with self.assertRaises(ValueError):synthesize(row['prompt'],caption,row['seed'],path)
                teacher.assert_not_called()

    def test_final_condition_metadata_and_finalized_inputs_are_rejected(self):
        for mode in ('final','finalized'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();plan,prior,out,_,rows=self.fixture(root)
                if mode=='final':
                    path=prior/'progress.json';r=json.loads(path.read_text());r['attempts'][0]['condition_id']=plan['conditions'][-1]['id'];path.write_text(json.dumps(r))
                else:(prior/'train.npz').write_bytes(b'already partitioned')
                with self.assertRaises(ValueError):cache.source_callback(plan,prior,synthesize=Mock())


if __name__=='__main__':unittest.main()
