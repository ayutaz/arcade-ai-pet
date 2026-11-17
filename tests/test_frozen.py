import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from arcade_ai_pet.features import split_for
from arcade_ai_pet.quantize import quantize_layer
from arcade_ai_pet.frozen import freeze_model, evaluate_frozen
from arcade_ai_pet.frozen import read_layers


def dataset(root,prefix):
    root.mkdir();records=[]
    for split in ('train','validation','test'):
        labels=[];features=[]
        for label in range(7):
            source=next(f'{prefix}/{label}/{i}' for i in range(10000) if split_for(f'{prefix}/{label}/{i}')==split)
            records.append(dict(source=source,id=hashlib.sha256(source.encode()).hexdigest(),split=split,label=label))
            vector=np.zeros(260,dtype=np.float32);vector[label]=4;features.append(vector);labels.append(label)
        np.savez(root/(split+'.npz'),x=np.array(features),y=np.array(labels))
    (root/'manifest.json').write_text(json.dumps(dict(feature_version='aligned-mfcc-v2',variants=1,records=records)))


def checkpoint(root):
    root.mkdir()
    layers=[quantize_layer(np.eye(7,260),np.zeros(7)),quantize_layer(np.eye(7),np.zeros(7))]
    np.savez(root/'weights.npz',**{f'{field}{i}':layer[field] for i,layer in enumerate(layers) for field in ('weight','bias','shift')})
    (root/'validation.json').write_text(json.dumps(dict(threshold_q10=0,selected_epoch=1,test_evaluated=False,
        validation_selected=dict(macro_f1=1,false_activation_rate=0))))


class FrozenTests(unittest.TestCase):
    def fixture(self,base):
        dataset(base/'development','development');dataset(base/'holdout','holdout');checkpoint(base/'checkpoint')
        return freeze_model(base/'checkpoint',base/'development',base/'frozen')

    def test_frozen_model_is_hashed_and_new_source_evaluation_does_not_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);model=self.fixture(base)
            result=evaluate_frozen(base/'frozen',base/'holdout',base/'report')
            self.assertEqual(result['macro_f1'],1);self.assertTrue(result['accepted_synthetic'])
            self.assertFalse(result['real_voice_tested']);self.assertFalse(result['device_tested'])
            self.assertEqual(result['weights_sha256'],model['weights_sha256'])
            self.assertEqual(result['source_overlap'],0);self.assertEqual(result['examples'],21)

    def test_reused_sources_tampered_weights_and_existing_outputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);self.fixture(base)
            with self.assertRaises(ValueError):evaluate_frozen(base/'frozen',base/'development',base/'no-report')
            with self.assertRaises(ValueError):freeze_model(base/'checkpoint',base/'development',base/'frozen')
            evaluate_frozen(base/'frozen',base/'holdout',base/'report')
            with self.assertRaises(ValueError):evaluate_frozen(base/'frozen',base/'holdout',base/'report')
            with (base/'frozen/weights.npz').open('ab') as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):evaluate_frozen(base/'frozen',base/'holdout',base/'no-report')
            self.assertFalse((base/'no-report').exists())

    def test_unacceptable_validation_cannot_be_frozen_as_a_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);dataset(base/'development','development');checkpoint(base/'checkpoint')
            (base/'checkpoint/validation.json').write_text(json.dumps(dict(threshold_q10=0,selected_epoch=1,test_evaluated=False,
                validation_selected=dict(macro_f1=.5,false_activation_rate=.1))))
            with self.assertRaises(ValueError):freeze_model(base/'checkpoint',base/'development',base/'frozen')

    def test_relabelled_source_ids_do_not_make_identical_voice_files_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);dataset(base/'development','development');dataset(base/'holdout','holdout');checkpoint(base/'checkpoint')
            for name in ('development','holdout'):
                path=base/name/'manifest.json';metadata=json.loads(path.read_text())
                metadata['records'][0]['sha256']='a'*64;path.write_text(json.dumps(metadata))
            freeze_model(base/'checkpoint',base/'development',base/'frozen')
            with self.assertRaises(ValueError):evaluate_frozen(base/'frozen',base/'holdout',base/'no-report')

    def test_reported_success_does_not_replace_independent_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);dataset(base/'development','development');checkpoint(base/'checkpoint')
            with np.load(base/'checkpoint/weights.npz',allow_pickle=False) as archive:
                values={key:archive[key] for key in archive.files}
            values['weight1'][:]=0;np.savez(base/'checkpoint/weights.npz',**values)
            with self.assertRaises(ValueError):freeze_model(base/'checkpoint',base/'development',base/'frozen')

    def test_model_array_header_is_bounded_before_numpy_allocation(self):
        import io,zipfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint(Path(tmp)/'checkpoint')
            path=Path(tmp)/'malformed.npz';header=io.BytesIO()
            np.lib.format.write_array_header_1_0(header,dict(descr='|i1',fortran_order=False,shape=(1000000000,260)))
            with zipfile.ZipFile(Path(tmp)/'checkpoint/weights.npz') as good,zipfile.ZipFile(path,'w') as archive:
                for name in good.namelist():archive.writestr(name,header.getvalue() if name=='weight0.npy' else good.read(name))
            with patch('arcade_ai_pet.frozen.np.load',side_effect=AssertionError('must reject before allocation')):
                with self.assertRaises(ValueError):read_layers(path)

    def test_wider_integer_model_is_accepted_only_within_24kib_shape_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for hidden in (88,91):
                path=root/(str(hidden)+'.npz')
                np.savez(path,weight0=np.zeros((hidden,260),dtype=np.int8),bias0=np.zeros(hidden,dtype=np.int32),shift0=np.int64(1),
                         weight1=np.zeros((7,hidden),dtype=np.int8),bias1=np.zeros(7,dtype=np.int32),shift1=np.int64(1))
                if hidden==88:
                    layers,size=read_layers(path);self.assertEqual(size,23884);self.assertEqual(layers[1]['weight'].shape,(7,88))
                else:
                    from unittest.mock import patch
                    with patch('arcade_ai_pet.frozen.np.load',side_effect=AssertionError('reject before payload load')):
                        with self.assertRaises(ValueError):read_layers(path)
