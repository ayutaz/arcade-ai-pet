"""点検済みの新発話だけをtrainへ追加し、旧評価をbyte単位で保持。"""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import training_extension as extension
from arcade_ai_pet.neural_corpus import generate_development,validate_plan
from arcade_ai_pet.features import split_for
from arcade_ai_pet.training import load_dataset
from arcade_ai_pet.native_evaluation import _examples
from test_mixed_corpus import fixture
import test_neural_corpus as neural_fixture
plan=neural_fixture.plan


class TrainingExtensionTests(unittest.TestCase):
    def fixture(self,root):
        base=root/'base';manifest=fixture(base,'base');original=plan()
        manifest['plan_sha256']=validate_plan(original);(base/'manifest.json').write_text(json.dumps(manifest))
        frozen=root/'reserved-plan.json';frozen.write_text(json.dumps(original))
        added=copy.deepcopy(original);added['conditions'][0].update(id='extra-01',caption='はっきりした落ち着いた成人の声。');added['conditions'][1].update(id='extra-02',caption='明るく柔らかい成人の声。')
        callbacks=neural_fixture.NeuralCorpusTests().provider();extra=root/'extra'
        generate_development(added,extra,synthesize=callbacks[0],recognize=callbacks[1],finalize=False)
        return base,extra,frozen,manifest

    def test_only_train_changes_and_original_evaluation_and_records_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();base,extra,frozen,manifest=self.fixture(root)
            before={name:(base/name).read_bytes() for name in ('manifest.json','validation.npz','test.npz')}
            result=extension.append_qualified_training(base,extra,root/'extended',reserved_plan=frozen)
            output=root/'extended';actual=json.loads((output/'manifest.json').read_text());data=load_dataset(output)
            self.assertEqual(actual['records'][:60],manifest['records']);self.assertEqual(result['speech_sources_added'],42)
            for name,raw in before.items():self.assertEqual((base/name).read_bytes(),raw)
            for name in ('validation.npz','test.npz'):self.assertEqual((output/name).read_bytes(),before[name])
            added=actual['records'][60:];self.assertEqual({r['split'] for r in added},{'train'})
            self.assertTrue(all(split_for(r['source'])=='train' and hashlib.sha256(r['source'].encode()).hexdigest()==r['id'] for r in added))
            self.assertTrue(all(r['origin_id']!=r['id'] for r in added));self.assertEqual(len(list(_examples(output))),102)
            self.assertEqual(len(data['train']['y']),len(load_dataset(base)['train']['y'])+42)
            self.assertFalse(result['final_conditions_used']);self.assertFalse(result['validation_changed'])
            self.assertEqual(actual['training_extensions'][0]['provider'],plan()['provider'])

    def test_final_incomplete_changed_or_finalized_extension_never_creates_output(self):
        for mode in ('final','incomplete','changed','finalized','different-reserved-plan'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();base,extra,frozen,manifest=self.fixture(root)
                if mode=='final':
                    p=extra/'qualified-sources.json';r=json.loads(p.read_text());r['role']='final';p.write_text(json.dumps(r))
                elif mode=='incomplete':
                    p=extra/'progress.json';r=json.loads(p.read_text());r['status']='incomplete';p.write_text(json.dumps(r))
                elif mode=='changed':
                    r=json.loads((extra/'qualified-sources.json').read_text())['records'][0];(extra/'sources'/(r['id']+'.wav')).write_bytes(b'changed')
                elif mode=='finalized':(extra/'train.npz').write_bytes(b'not an unpartitioned source corpus')
                else:r=json.loads(frozen.read_text());r['seed']+=1;frozen.write_text(json.dumps(r))
                with self.assertRaises(ValueError):extension.append_qualified_training(base,extra,root/'output',reserved_plan=frozen)
                self.assertFalse((root/'output').exists())

    def test_changed_reserved_conditions_and_duplicate_speech_are_rejected(self):
        for mode in ('reserved-caption','duplicate-wave','failed-audit'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();base,extra,frozen,manifest=self.fixture(root)
                qpath=extra/'qualified-sources.json';q=json.loads(qpath.read_text());ppath=extra/'progress.json';progress=json.loads(ppath.read_text())
                if mode=='reserved-caption':
                    p=extra/'plan.json';v=json.loads(p.read_text());v['conditions'][-1]['caption']='別の未使用条件';p.write_text(json.dumps(v))
                    q['plan_sha256']=validate_plan(v);progress['plan_sha256']=q['plan_sha256']
                elif mode=='failed-audit':progress['attempts'][0]['content_check_passed']=False
                else:
                    row=q['records'][0];raw=(base/'sources'/(manifest['records'][0]['id']+'.wav')).read_bytes();(extra/'sources'/(row['id']+'.wav')).write_bytes(raw)
                    row['sha256']=hashlib.sha256(raw).hexdigest()
                    for audit in progress['attempts']:
                        if audit['id']==row['id']:audit['wave_sha256']=row['sha256']
                qpath.write_text(json.dumps(q));ppath.write_text(json.dumps(progress))
                with self.assertRaises(ValueError):extension.append_qualified_training(base,extra,root/'output',reserved_plan=frozen)
                self.assertFalse((root/'output').exists())

    def test_invalid_base_wave_is_rejected_before_output_creation(self):
        for mode in ('changed','symlink'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();base,extra,frozen,manifest=self.fixture(root)
                file=base/'sources'/(manifest['records'][0]['id']+'.wav')
                if mode=='changed':file.write_bytes(b'changed')
                else:
                    target=root/'same-wave.wav';target.write_bytes(file.read_bytes());file.unlink();file.symlink_to(target)
                with self.assertRaises(ValueError):extension.append_qualified_training(base,extra,root/'output',reserved_plan=frozen)
                self.assertFalse((root/'output').exists())

    def test_malformed_base_id_is_rejected_before_reading_an_external_wave(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();base,extra,frozen,manifest=self.fixture(root)
            original=base/'sources'/(manifest['records'][0]['id']+'.wav');external=root/'outside.wav';external.write_bytes(original.read_bytes())
            manifest['records'][0]['id']=str(external)[:-4];(base/'manifest.json').write_text(json.dumps(manifest))
            original_sha=extension._sha
            def bounded_sha(path):
                if Path(path)==external:raise AssertionError('external source was read before identity validation')
                return original_sha(path)
            with patch.object(extension,'_sha',side_effect=bounded_sha):
                with self.assertRaises(ValueError):extension.append_qualified_training(base,extra,root/'output',reserved_plan=frozen)
            self.assertFalse((root/'output').exists())

    def test_existing_output_and_unreviewed_provider_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();base,extra,frozen,_=self.fixture(root)
            with self.assertRaises(ValueError):extension.append_qualified_training(base,extra,base,reserved_plan=frozen)
            p=extra/'plan.json';r=json.loads(p.read_text());r['provider']['model_sha256']='6'*64;p.write_text(json.dumps(r))
            with self.assertRaises(ValueError):extension.append_qualified_training(base,extra,root/'output',reserved_plan=frozen)
            self.assertFalse((root/'output').exists())


if __name__=='__main__':unittest.main()
