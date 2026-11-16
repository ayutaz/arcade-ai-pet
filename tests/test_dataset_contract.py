import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from arcade_ai_pet.features import split_for
from arcade_ai_pet.training import load_dataset


class DatasetContractTests(unittest.TestCase):
    def fixture(self,root):
        records=[]
        for split in ('train','validation','test'):
            index=next(i for i in range(1000) if split_for(f'fixture-{i}')==split)
            source=f'fixture-{index}'
            records.append(dict(source=source,id=hashlib.sha256(source.encode()).hexdigest(),split=split,label=6))
            np.savez(root/(split+'.npz'),x=np.zeros((1,260),dtype=np.float32),y=np.array([6]))
        (root/'manifest.json').write_text(json.dumps(dict(variants=1,records=records)))
        return records

    def test_source_identity_split_and_feature_count_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);records=self.fixture(root)
            data=load_dataset(root);self.assertEqual(len(data['train']['y']),1)
            records[1]['id']=records[0]['id']
            (root/'manifest.json').write_text(json.dumps(dict(variants=1,records=records)))
            with self.assertRaises(ValueError):load_dataset(root)

    def test_nonfinite_features_and_wrong_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            np.savez(root/'test.npz',x=np.full((1,260),np.nan),y=np.array([6]))
            with self.assertRaises(ValueError):load_dataset(root)
            np.savez(root/'test.npz',x=np.zeros((1,260)),y=np.array([0]))
            with self.assertRaises(ValueError):load_dataset(root)
