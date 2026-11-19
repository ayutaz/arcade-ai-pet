"""平均スペクトルの追加で元archive/保留/加工条件を変えない。"""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import logmel_arrays,caption_holdout,envelope_frontend,logmel_frontend
from test_mixed_corpus import fixture as speech_fixture
from test_caption_holdout import fixture as held_fixture,patches,SEED

MODE='context-logmel64-envelope'

class EnvelopeArraysTests(unittest.TestCase):
    def test_real_legacy_arrays_keep_prefix_labels_profiles_and_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()/'data';speech_fixture(root,'envelope')
            before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.npz')}
            old=logmel_arrays.make_arrays(root,'hp160',noise_replicas=2,hum_replicas=2)
            new=logmel_arrays.make_arrays(root,'hp160',frontend=MODE,noise_replicas=2,hum_replicas=2)
            for split in old:
                self.assertEqual(new[split]['x'].shape[1],2145)
                np.testing.assert_array_equal(new[split]['x'][:,:2113],old[split]['x'])
                np.testing.assert_array_equal(new[split]['y'],old[split]['y']);self.assertEqual(new[split]['profiles'],old[split]['profiles'])
            self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.npz')})

    def test_holdout_routes_new_frontend_without_changing_source_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();_,_,data=held_fixture(root);selection=caption_holdout.select_holdout(root,seed=SEED)
            # Fresh generators for both reconstructions; no final/test source is added.
            with patches(data),patch.object(logmel_frontend,'features',return_value=np.zeros(2113,dtype=np.float32)):
                old=caption_holdout.make_arrays(root,selection=selection,frontend='context-logmel64',noise_replicas=1,hum_replicas=1)
            with patches(data),patch.object(envelope_frontend,'features',return_value=np.zeros(2145,dtype=np.float32)) as feature:
                new=caption_holdout.make_arrays(root,selection=selection,frontend=MODE,noise_replicas=1,hum_replicas=1)
            self.assertTrue(feature.called)
            for split in old:
                self.assertEqual(new[split]['x'].shape[1],2145)
                for key in ('source_ids','profiles','condition_ids','variants'):self.assertEqual(new[split][key],old[split][key])
                np.testing.assert_array_equal(new[split]['y'],old[split]['y'])
            self.assertFalse(set(new['train']['source_ids'])&set(new['held_validation']['source_ids']))

    def test_final_and_foreign_frontend_still_refuse_before_pcm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();(root/'manifest.json').write_text(json.dumps(dict(role='final')))
            with patch.object(logmel_arrays,'_examples',side_effect=AssertionError('PCM reached')):
                with self.assertRaises(ValueError):logmel_arrays.make_arrays(root,'hp160',frontend=MODE)
                with self.assertRaises(ValueError):logmel_arrays.make_arrays(root,'hp160',frontend='foreign')

if __name__=='__main__':unittest.main()
