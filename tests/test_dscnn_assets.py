"""先行契約: 6層の整数assetを安全に読み、再生成したheaderだけを使う。"""
from pathlib import Path
import hashlib,json,tempfile,unittest
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import dscnn_assets as assets
from arcade_ai_pet.dscnn_model import scores
from arcade_ai_pet.class_rejection import decisions
from test_dscnn_model import fixture


def model(path):
    layers=fixture();np.savez(path,**{f'{key}{i}':layer[key] for i,layer in enumerate(layers) for key in ('weight','bias','shift')});return layers


class DscnnAssetsTests(unittest.TestCase):
    def test_numeric_roundtrip_and_generated_headers_bind_the_same_model_and_five_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();layers=model(root/'weights.npz');table=[1,2,3,4,5]
            result=assets.export_assets(root/'weights.npz',table,root/'assets');actual,size=assets.read_layers(root/'assets/weights.npz')
            self.assertEqual(size,23391);self.assertEqual(result['numeric_bytes'],23411);self.assertFalse(result['frozen'])
            self.assertEqual(result['runtime'],assets.RUNTIME);self.assertEqual(result['input_contract']['features'],2113)
            self.assertEqual(set(result['header_sha256']),{'logmel_tables.h','logmel_model.h'})
            for a,b in zip(layers,actual,strict=True):
                for key in a:np.testing.assert_array_equal(a[key],b[key])
            rendered=assets.render(layers,table)
            for name,text in rendered.items():
                self.assertEqual((root/'assets'/name).read_text(),text)
                self.assertEqual(result['header_sha256'][name],hashlib.sha256(text.encode()).hexdigest())
            self.assertIn('pet_logmel_layers[6]',rendered['logmel_model.h'])

    def test_invalid_archive_headers_members_and_types_refuse_before_numpy_load(self):
        for kind in ('missing','dtype','shape','shift_type','extra'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();layers=fixture();arrays={f'{key}{i}':layer[key] for i,layer in enumerate(layers) for key in ('weight','bias','shift')}
                if kind=='missing':del arrays['bias5']
                if kind=='dtype':arrays['weight0']=arrays['weight0'].astype(float)
                if kind=='shape':arrays['weight5']=arrays['weight5'][:,:865]
                if kind=='shift_type':arrays['shift0']=arrays['shift0'].astype(np.int64)
                if kind=='extra':arrays['unexpected']=np.array([1])
                np.savez(root/'bad.npz',**arrays)
                with patch.object(assets.np,'load',side_effect=AssertionError('allocation reached')):
                    with self.assertRaises(ValueError):assets.read_layers(root/'bad.npz')

    def test_bad_shifts_and_thresholds_refuse_without_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();model(root/'weights.npz')
            for table in ([0]*4,[0,0,0,0,True],[-1,0,0,0,0],[4097,0,0,0,0]):
                with self.assertRaises(ValueError):assets.export_assets(root/'weights.npz',table,root/'assets')
                self.assertFalse((root/'assets').exists())
            layers=fixture();layers[0]['shift'][0]=17
            with self.assertRaises(ValueError):assets.render(layers,[0]*5)

    def test_symlinks_and_existing_output_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();model(root/'weights.npz');(root/'link.npz').symlink_to(root/'weights.npz')
            with self.assertRaises(ValueError):assets.read_layers(root/'link.npz')
            (root/'output').mkdir()
            with self.assertRaises(ValueError):assets.export_assets(root/'weights.npz',[0]*5,root/'output')

    def test_asset_runtime_scores_and_extract_match_existing_c_and_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();layers=model(root/'weights.npz');table=[1,2,3,4,5]
            assets.export_assets(root/'weights.npz',table,root/'assets')
            with assets.DscnnNativeAssets(root/'assets') as engine:
                x=np.zeros(2113,np.float32);label,logits=engine.predict(x);expected=scores(x[None,:],layers)
                np.testing.assert_array_equal(logits,expected[0]);self.assertEqual(label,int(decisions(expected,np.array(table,np.int32))[0]))
                self.assertEqual(engine.scratch_bytes,26672)
                np.testing.assert_array_equal(engine.extract(np.zeros(32000,np.int16)),x)
                np.testing.assert_array_equal(engine.extract(np.zeros(32000,np.int16),banked=True),x)

    def test_forged_header_refuses_even_if_supplied_metadata_hash_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();model(root/'weights.npz');assets.export_assets(root/'weights.npz',[0]*5,root/'assets')
            p=root/'assets/logmel_model.h';p.write_text('#include "/tmp/not-trusted"\n');meta_path=root/'assets/decision-metadata.json';meta=json.loads(meta_path.read_text());meta['header_sha256']['logmel_model.h']=hashlib.sha256(p.read_bytes()).hexdigest();meta_path.write_text(json.dumps(meta))
            with patch.object(assets.subprocess,'run',side_effect=AssertionError('compiler reached')):
                with self.assertRaises(ValueError):assets.DscnnNativeAssets(root/'assets')

    def test_changed_numeric_metadata_refuses_before_compiling(self):
        for key,bad in (('runtime','legacy'),('numeric_bytes',1),('weights_sha256','0'*64),('frozen',True)):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();model(root/'weights.npz');assets.export_assets(root/'weights.npz',[0]*5,root/'assets')
                p=root/'assets/decision-metadata.json';meta=json.loads(p.read_text());meta[key]=bad;p.write_text(json.dumps(meta))
                with patch.object(assets.subprocess,'run',side_effect=AssertionError('compiler reached')):
                    with self.assertRaises(ValueError):assets.DscnnNativeAssets(root/'assets')

if __name__=='__main__':unittest.main()
