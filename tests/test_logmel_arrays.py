"""log-melへの変更で元の分割・ラベル・profile・archiveを変えない。"""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import logmel_arrays as arrays
from arcade_ai_pet.cmn_arrays import make_arrays as previous
from test_mixed_corpus import fixture


class LogmelArraysTests(unittest.TestCase):
    def test_real_reconstruction_preserves_labels_profiles_and_original_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset=Path(tmp).resolve()/'data';fixture(dataset,'logmel');before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dataset.glob('*.npz')}
            options=dict(noise_replicas=2,noise_mode='fresh',hum_replicas=2)
            old=previous(dataset,'hp160',**options);new=arrays.make_arrays(dataset,'hp160',**options)
            self.assertEqual(set(new),{'train','validation'})
            for split in new:
                np.testing.assert_array_equal(new[split]['y'],old[split]['y']);self.assertEqual(new[split]['profiles'],old[split]['profiles'])
                self.assertEqual(new[split]['x'].shape[1],2113);self.assertTrue(np.isfinite(new[split]['x']).all())
                self.assertTrue((new[split]['x'][:,2048:2112]<=0).all())
            self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dataset.glob('*.npz')})

    def test_final_or_wrong_profile_refuses_before_pcm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();(root/'manifest.json').write_text(json.dumps(dict(role='final')))
            with patch.object(arrays,'_examples',side_effect=AssertionError('PCM reached')):
                with self.assertRaises(ValueError):arrays.make_arrays(root,'hp160')
                with self.assertRaises(ValueError):arrays.make_arrays(root,'hp160',frontend='context-duration-cmn')
                with self.assertRaises(ValueError):arrays.make_arrays(root,'none')


if __name__=='__main__':unittest.main()
