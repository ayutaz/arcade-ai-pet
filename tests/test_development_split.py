"""学習前の開発データだけを元発話単位で層化する。"""
import hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
from arcade_ai_pet.development_split import prepare_development
from arcade_ai_pet.neural_corpus import generate_development
from arcade_ai_pet.training import load_dataset
from arcade_ai_pet.native_evaluation import _examples
import test_neural_corpus as fixture


def dataset(path,conditions=3):
    plan=fixture.plan()
    if conditions==3:plan['conditions'].append(dict(id='dev-03',role='development',caption='中くらいの声で話す。'))
    synthesize,recognize,_=fixture.NeuralCorpusTests().provider()
    generate_development(plan,path,synthesize=synthesize,recognize=recognize)


class DevelopmentSplitTests(unittest.TestCase):
    def test_every_class_has_source_disjoint_splits_and_reconstructs_with_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();base=root/'base';dataset(base);before=(base/'manifest.json').read_bytes()
            result=prepare_development(base,root/'prepared');self.assertTrue(result['class_coverage_complete'])
            manifest=json.loads((root/'prepared/manifest.json').read_text());original=json.loads(before)
            self.assertEqual(manifest['origin_manifest_sha256'],hashlib.sha256(before).hexdigest())
            speech=[r for r in manifest['records'] if 'sha256' in r];old=[r for r in original['records'] if 'sha256' in r]
            self.assertEqual({r['origin_id'] for r in speech},{r['id'] for r in old})
            self.assertEqual({r['sha256'] for r in speech},{r['sha256'] for r in old})
            self.assertEqual(len(list(_examples(root/'prepared'))),263)
            for split,arrays in load_dataset(root/'prepared').items():self.assertEqual(set(arrays['y']),set(range(7)))
            self.assertEqual((base/'manifest.json').read_bytes(),before)
            ids={split:{r.get('origin_id',r['id']) for r in manifest['records'] if r['split']==split} for split in ('train','validation','test')}
            self.assertFalse(ids['train']&ids['validation'] or ids['train']&ids['test'] or ids['validation']&ids['test'])

    def test_repeat_is_deterministic_without_overwriting_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();dataset(root/'base');prepare_development(root/'base',root/'a');prepare_development(root/'base',root/'b')
            self.assertEqual((root/'a/manifest.json').read_bytes(),(root/'b/manifest.json').read_bytes())
            with self.assertRaises(ValueError):prepare_development(root/'base',root/'a')
            for split in ('train','validation','test'):
                a=load_dataset(root/'a')[split];b=load_dataset(root/'b')[split];np.testing.assert_array_equal(a['x'],b['x'])

    def test_final_role_insufficient_command_sources_and_changed_audio_reject_before_output(self):
        for mode in ('final','insufficient','changed'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();base=root/'base';dataset(base,conditions=2 if mode=='insufficient' else 3)
                path=base/'manifest.json';m=json.loads(path.read_text())
                if mode=='final':m['role']='final';path.write_text(json.dumps(m))
                if mode=='changed':(base/'sources'/(m['records'][0]['id']+'.wav')).write_bytes(b'changed')
                with self.assertRaises(ValueError):prepare_development(base,root/'prepared')
                self.assertFalse((root/'prepared').exists())


if __name__=='__main__':unittest.main()
