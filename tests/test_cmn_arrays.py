"""既存の元発話/加工/比重/validationを変えず、別の特徴契約にする。"""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import cmn_arrays as cmn
from arcade_ai_pet.cmn_frontend import center
from arcade_ai_pet.duration_arrays import make_arrays as previous
from test_mixed_corpus import fixture


class CmnArraysTests(unittest.TestCase):
    def test_only_features_change_while_source_files_labels_and_profiles_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset=Path(tmp).resolve()/'data';fixture(dataset,'cmn')
            before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dataset.glob('*.npz')}
            options=dict(noise_replicas=2,hum_replicas=2,noise_mode='fresh')
            old=previous(dataset,'hp160',frontend='context-duration',**options)
            actual=cmn.make_arrays(dataset,'hp160',frontend='context-duration-cmn',**options)
            self.assertEqual(set(actual),{'train','validation'})
            for split in actual:
                np.testing.assert_array_equal(actual[split]['x'],center(old[split]['x']))
                np.testing.assert_array_equal(actual[split]['x'][:,240:],old[split]['x'][:,240:])
                np.testing.assert_array_equal(actual[split]['y'],old[split]['y']);self.assertEqual(actual[split]['profiles'],old[split]['profiles'])
            self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dataset.glob('*.npz')})

    def test_final_or_foreign_frontend_is_rejected_before_calling_the_source_builder(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset=Path(tmp).resolve();(dataset/'manifest.json').write_text(json.dumps(dict(role='final')))
            with patch.object(cmn,'original',side_effect=AssertionError('final source builder reached')):
                with self.assertRaises(ValueError):cmn.make_arrays(dataset,'hp160')
            with patch.object(cmn,'original',side_effect=AssertionError('foreign frontend builder reached')):
                with self.assertRaises(ValueError):cmn.make_arrays(dataset,'hp160',frontend='context-duration')
                with self.assertRaises(ValueError):cmn.make_arrays(dataset,'none')

    def test_test_split_is_never_silently_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset=Path(tmp).resolve();(dataset/'manifest.json').write_text(json.dumps(dict(role='development')))
            arrays={s:dict(x=np.zeros((1,261),dtype=np.float32),y=np.array([6]),profiles=['clean']) for s in ('train','validation','test')}
            with patch.object(cmn,'original',return_value=arrays):
                with self.assertRaises(ValueError):cmn.make_arrays(dataset,'hp160')


if __name__=='__main__':unittest.main()
