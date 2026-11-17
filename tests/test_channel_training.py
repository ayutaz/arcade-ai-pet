import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import channel_training as training
from arcade_ai_pet.channel_robustness import PROFILES


class ChannelTrainingTests(unittest.TestCase):
    def test_level_preserves_zero_and_uses_full_int16_range_without_wrapping(self):
        zero=np.zeros(32000,dtype=np.int16)
        np.testing.assert_array_equal(training.level(zero),zero)
        pcm=zero.copy();pcm[100]=-32768;pcm[101]=32767
        saved=pcm.copy();result=training.level(pcm)
        self.assertEqual(result.dtype,np.int16);self.assertEqual(result.min(),-4096)
        self.assertLessEqual(result.max(),4096);np.testing.assert_array_equal(pcm,saved)
        pcm[100]=-2;pcm[101]=2
        result=training.level(pcm);self.assertEqual(result[100],-4096);self.assertEqual(result[101],4096)
        for bad in (np.zeros(32000),np.zeros(31999,dtype=np.int16)):
            with self.assertRaises(ValueError):training.level(bad)

    def test_training_arrays_keep_source_splits_and_exclude_original_test_from_fitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);records=[];examples=[]
            for label,split in enumerate(('train','validation','test')):
                source='unit/'+split
                records.append(dict(source=source,id=hashlib.sha256(source.encode()).hexdigest(),split=split,label=label,sha256=str(label)*64))
                pcm=np.zeros(32000,dtype=np.int16);pcm[4000:12000]=(1000*np.sin(np.arange(8000)*.1)).astype(np.int16)
                examples.append((pcm,label,np.zeros(260,dtype=np.float32)))
            (root/'manifest.json').write_text(json.dumps(dict(variants=1,records=records)))
            with patch.object(training,'_examples',return_value=iter(examples)):
                data=training.make_arrays(root,'hp80')
            self.assertEqual(set(data),{'train','validation'})
            for split,label in [('train',0),('validation',1)]:
                self.assertEqual(data[split]['x'].shape,(len(PROFILES),260))
                self.assertEqual(data[split]['y'].tolist(),[label]*len(PROFILES))
                self.assertEqual(data[split]['profiles'],list(PROFILES))
                self.assertTrue(np.isfinite(data[split]['x']).all())
                self.assertGreater(np.linalg.norm(data[split]['x'][1]),1)

    def test_unknown_conditioner_cannot_create_training_data(self):
        with self.assertRaises(ValueError):training.make_arrays(Path('absent'),'unreviewed')
