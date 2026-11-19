"""既存の元発話・split・profileを変えず、261特徴を再構築する。"""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import duration_arrays as duration
from arcade_ai_pet.channel_training import make_arrays as previous
from arcade_ai_pet.training import load_dataset
from arcade_ai_pet.context_arrays import make_arrays as context_arrays
from test_mixed_corpus import fixture


class DurationArraysTests(unittest.TestCase):
    def test_existing_features_and_validation_profiles_are_preserved_without_test_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();dataset=root/'data';fixture(dataset,'duration')
            before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dataset.glob('*.npz')}
            options=dict(noise_replicas=2,hum_replicas=2,noise_mode='fresh')
            old=previous(dataset,'hp160',frontend='adaptive',**options)
            actual=duration.make_arrays(dataset,'hp160',frontend='duration',**options)
            data=load_dataset(dataset)
            self.assertEqual(set(actual),{'train','validation'})
            for split in actual:
                np.testing.assert_array_equal(actual[split]['x'][:,:260],old[split]['x'])
                np.testing.assert_array_equal(actual[split]['y'],old[split]['y'])
                self.assertEqual(actual[split]['profiles'],old[split]['profiles'])
                self.assertEqual(actual[split]['x'].shape[1],261)
            self.assertEqual(len(actual['validation']['y']),8*len(data['validation']['y']))
            self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dataset.glob('*.npz')})

    def test_final_corpus_and_foreign_frontend_are_rejected_before_pcm_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();fixture(root/'data','duration')
            path=root/'data/manifest.json';meta=json.loads(path.read_text());meta['role']='final';path.write_text(json.dumps(meta))
            with patch.object(duration,'_examples',side_effect=AssertionError('final PCM read')):
                with self.assertRaises(ValueError):duration.make_arrays(root/'data','hp160')
            with self.assertRaises(ValueError):duration.make_arrays(root/'data','hp160',frontend='adaptive')

    def test_context_duration_keeps_context_spectra_and_the_same_unpadded_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset=Path(tmp).resolve()/'data';fixture(dataset,'context-duration')
            options=dict(noise_replicas=2,hum_replicas=2,noise_mode='fresh')
            old=context_arrays(dataset,'hp160',frontend='context',**options)
            timing=duration.make_arrays(dataset,'hp160',frontend='duration',**options)
            actual=duration.make_arrays(dataset,'hp160',frontend='context-duration',**options)
            for split in ('train','validation'):
                np.testing.assert_array_equal(actual[split]['x'][:,:260],old[split]['x'])
                np.testing.assert_array_equal(actual[split]['x'][:,-1],timing[split]['x'][:,-1])
                np.testing.assert_array_equal(actual[split]['y'],old[split]['y'])
                self.assertEqual(actual[split]['profiles'],old[split]['profiles'])

    def test_context_duration_still_rejects_final_data_before_reading_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();fixture(root/'data','context-duration')
            path=root/'data/manifest.json';meta=json.loads(path.read_text());meta['role']='final';path.write_text(json.dumps(meta))
            with patch.object(duration,'_examples',side_effect=AssertionError('final PCM read')):
                with self.assertRaises(ValueError):duration.make_arrays(root/'data','hp160',frontend='context-duration')


if __name__=='__main__':unittest.main()
