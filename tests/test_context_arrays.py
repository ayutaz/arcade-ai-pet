import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import context_arrays as arrays


class ContextArraysTests(unittest.TestCase):
    def test_only_training_is_augmented_and_test_pcm_is_not_sent_to_context_extractor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);records=[dict(id=hashlib.sha256(split.encode()).hexdigest(),source=split,split=split,label=label,sha256='a'*64) for label,split in enumerate(('train','validation','test'))]
            (root/'manifest.json').write_text(json.dumps(dict(variants=1,records=records)))
            pcm=np.rint(1000*np.sin(np.arange(32000)*.1)).astype(np.int16)
            examples=[(pcm,0,None),(pcm,1,None),(object(),2,None)]
            with patch.object(arrays,'_examples',return_value=iter(examples)):
                result=arrays.make_arrays(root,'hp160',frontend='context',noise_replicas=5,noise_mode='fresh',hum_replicas=3)
            self.assertEqual(len(result['train']['y']),14);self.assertEqual(len(result['validation']['y']),8)
            self.assertEqual(set(result['train']['y']),{0});self.assertEqual(set(result['validation']['y']),{1})
            self.assertEqual(result['train']['x'].shape,(14,260));self.assertTrue(np.isfinite(result['train']['x']).all())
            with self.assertRaises(ValueError):arrays.make_arrays(root,'hp160',frontend='adaptive')


if __name__=='__main__':unittest.main()
