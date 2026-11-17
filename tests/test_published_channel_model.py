import hashlib,json,unittest
from pathlib import Path
import numpy as np
from arcade_ai_pet.channel_native import ChannelNativeModel
from arcade_ai_pet.frozen import read_layers

ROOT=Path(__file__).resolve().parents[1]


class PublishedChannelModelTests(unittest.TestCase):
    def test_channel_asset_identity_license_and_resource_budget(self):
        base=ROOT/'models/kws-v3';metadata=json.loads((base/'channel-metadata.json').read_text())
        layers,size=read_layers(base/'weights.npz')
        self.assertEqual(size,23884);self.assertLessEqual(size,24576)
        self.assertEqual(layers[0]['weight'].shape,(88,260))
        self.assertEqual(metadata['weights_sha256'],hashlib.sha256((base/'weights.npz').read_bytes()).hexdigest())
        self.assertEqual(metadata['threshold_q10'],995);self.assertEqual(metadata['runtime'],'channel-adaptive-v1')
        self.assertEqual(metadata['frozen_sha256'],hashlib.sha256((base/'channel-frozen.json').read_bytes()).hexdigest())
        self.assertTrue((base/'LICENSE.txt').is_file());self.assertTrue((base/'model-card.md').is_file())
        self.assertFalse((base/'metadata.json').exists())

    def test_frozen_synthetic_conformance_vectors_match_native_integer_engine(self):
        base=ROOT/'models/kws-v3'
        with np.load(base/'conformance.npz',allow_pickle=False) as saved:
            x=saved['x'];expected=saved['logits'];labels=saved['prediction']
        self.assertEqual(x.shape,(56,260));self.assertEqual(x.dtype,np.float32)
        self.assertEqual(expected.shape,(56,7));self.assertEqual(labels.shape,(56,))
        with ChannelNativeModel(base) as model:
            for features,logits,label in zip(x,expected,labels,strict=True):
                actual,actual_logits=model.predict(features)
                np.testing.assert_array_equal(actual_logits,logits);self.assertEqual(actual,int(label))
