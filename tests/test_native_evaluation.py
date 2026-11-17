import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import wave
import numpy as np
from arcade_ai_pet.data import augment,trim_silence
from arcade_ai_pet.features import features,split_for
from arcade_ai_pet.frozen import freeze_model
from arcade_ai_pet.native import export_assets
from arcade_ai_pet.native_evaluation import NativeModel,evaluate_native
from test_frozen import dataset,checkpoint


def corpus(root):
    root.mkdir();(root/'sources').mkdir();records=[]
    x={split:[] for split in ('train','validation','test')};y={split:[] for split in x}
    for label in range(7):
        split=('train','validation','test')[label%3]
        prefix='native-noise' if label==6 else f'unit-voice/{label}'
        source=next(f'{prefix}/{i}' for i in range(200) if split_for(f'{prefix}/{i}')==split)
        ident=hashlib.sha256(source.encode()).hexdigest()
        if label==6:
            seed=int.from_bytes(hashlib.sha256(source.encode()).digest()[:8],'little')
            index=int(source.rsplit('/',1)[1]);rng=np.random.default_rng(seed)
            pcm=np.clip(rng.normal(0,0 if index%4==0 else rng.uniform(1,200),32000),-32767,32767).astype(np.int16)
            extra=dict(pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest())
        else:
            raw=(3000*np.sin(np.arange(8000)*(.07+label*.04))).astype(np.int16)
            file=root/'sources'/(ident+'.wav')
            with wave.open(str(file),'wb') as stream:
                stream.setparams((1,2,16000,0,'NONE','not compressed'));stream.writeframes(raw.astype('<i2').tobytes())
            pcm=augment(trim_silence(raw),int(ident[:8],16));extra=dict(sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        records.append(dict(source=source,id=ident,split=split,label=label,**extra));x[split].append(features(pcm));y[split].append(label)
    for split in x:np.savez(root/(split+'.npz'),x=np.array(x[split],dtype=np.float32),y=np.array(y[split],dtype=np.int64))
    (root/'manifest.json').write_text(json.dumps(dict(feature_version='aligned-mfcc-v2',variants=1,records=records)))


class NativeEvaluationTests(unittest.TestCase):
    def fixture(self,base):
        dataset(base/'development','development');checkpoint(base/'checkpoint')
        freeze_model(base/'checkpoint',base/'development',base/'frozen')
        export_assets(base/'frozen',base/'assets');shutil.copyfile(base/'frozen/weights.npz',base/'assets/weights.npz')

    def test_entire_synthetic_corpus_is_reconstructed_and_compared(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);self.fixture(base);corpus(base/'corpus')
            result=evaluate_native(base/'assets',base/'corpus',base/'report')
            self.assertEqual(result['examples'],7);self.assertTrue(result['native_matches_python'])
            self.assertEqual(result['exact_integer_score_matches'],7)
            self.assertLessEqual(result['scratch_bytes'],24576);self.assertFalse(result['device_tested'])

    def test_changed_header_or_waveform_is_rejected_without_ready_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);self.fixture(base);corpus(base/'corpus')
            original=(base/'assets/pet_model.h').read_text()
            original_metadata=(base/'assets/metadata.json').read_text()
            changed='#include "/tmp/foreign"';(base/'assets/pet_model.h').write_text(changed)
            metadata=json.loads(original_metadata);metadata['asset_sha256']['pet_model.h']=hashlib.sha256(changed.encode()).hexdigest()
            (base/'assets/metadata.json').write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):NativeModel(base/'assets')
            (base/'assets/pet_model.h').write_text(original)
            (base/'assets/metadata.json').write_text(original_metadata)
            file=next((base/'corpus/sources').glob('*.wav'));file.write_bytes(b'changed')
            with self.assertRaises(ValueError):evaluate_native(base/'assets',base/'corpus',base/'report')
            self.assertFalse((base/'report').exists())

    def test_existing_native_integer_engine_matches_python_for_88_hidden_units(self):
        from arcade_ai_pet.quantize import quantize_layer
        from arcade_ai_pet.training import scores
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frozen=root/'frozen';frozen.mkdir()
            layers=[quantize_layer(np.eye(88,260),np.zeros(88)),quantize_layer(np.eye(7,88),np.zeros(7))]
            np.savez(frozen/'weights.npz',**{f'{field}{i}':l[field] for i,l in enumerate(layers) for field in ('weight','bias','shift')})
            digest=hashlib.sha256((frozen/'weights.npz').read_bytes()).hexdigest()
            (frozen/'frozen.json').write_text(json.dumps(dict(weights_sha256=digest,threshold_q10=0,feature_version='aligned-mfcc-v2')))
            export_assets(frozen,root/'assets');shutil.copyfile(frozen/'weights.npz',root/'assets/weights.npz')
            with NativeModel(root/'assets') as model:
                x=np.zeros(260,dtype=np.float32);x[3]=4
                label,logits=model.predict(x)
                self.assertEqual(label,3);self.assertEqual(model.weights_bytes,23884)
                np.testing.assert_array_equal(logits,scores(x[None,:],layers)[0])
