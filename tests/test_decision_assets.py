"""261入力の数値形式だけを許可し、旧形式とCコードを混ぜない。"""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import decision_assets as assets
from arcade_ai_pet.frozen import read_layers as legacy_read


def model(path,hidden=7):
    rng=np.random.default_rng(718)
    layers=[dict(weight=rng.integers(-3,4,(hidden,261),dtype=np.int8),bias=rng.integers(-1000,1001,hidden,dtype=np.int32),shift=4),
            dict(weight=rng.integers(-3,4,(7,hidden),dtype=np.int8),bias=rng.integers(-1000,1001,7,dtype=np.int32),shift=3)]
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path,**{f'{field}{i}':r[field] for i,r in enumerate(layers) for field in ('weight','bias','shift')})
    return layers


class DecisionAssetsTests(unittest.TestCase):
    def test_new_width_is_explicit_and_numeric_budget_includes_threshold_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();layers=model(root/'weights.npz',90)
            actual,size=assets.read_layers(root/'weights.npz')
            self.assertEqual(actual[0]['weight'].shape,(90,261));self.assertEqual(size,24516)
            with self.assertRaises(ValueError):legacy_read(root/'weights.npz')
            meta=assets.export_assets(root/'weights.npz',[0,1,2,3,4],root/'assets')
            self.assertEqual(meta['numeric_bytes'],24536);self.assertFalse(meta['frozen'])

    def test_invalid_architecture_or_dtype_is_rejected_before_numpy_allocation(self):
        for mode in ('legacy-width','too-wide','too-deep','float','shift','object'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();layers=model(root/'weights.npz')
                if mode=='legacy-width':layers[0]['weight']=layers[0]['weight'][:,:260]
                elif mode=='too-wide':layers[0]['weight']=np.zeros((91,261),dtype=np.int8)
                elif mode=='too-deep':layers[1]['weight']=np.zeros((8,7),dtype=np.int8)
                elif mode=='float':layers[0]['weight']=layers[0]['weight'].astype(float)
                elif mode=='shift':layers[0]['shift']=17
                else:layers[0]['weight']=np.array([object()],dtype=object)
                np.savez(root/'weights.npz',**{f'{field}{i}':r[field] for i,r in enumerate(layers) for field in ('weight','bias','shift')})
                with patch.object(assets.np,'load',side_effect=AssertionError('allocation before header validation')):
                    with self.assertRaises(ValueError):assets.read_layers(root/'weights.npz')

    def test_export_is_deterministic_and_refuses_changed_tables_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();model(root/'weights.npz')
            a=assets.export_assets(root/'weights.npz',[977,2531,1780,181,768],root/'a')
            assets.export_assets(root/'weights.npz',[977,2531,1780,181,768],root/'b')
            for name in ('feature_tables.h','decision_model.h','decision-metadata.json','weights.npz'):
                self.assertEqual((root/'a'/name).read_bytes(),(root/'b'/name).read_bytes())
            self.assertEqual(a['input_contract']['features'],261)
            with self.assertRaises(ValueError):assets.export_assets(root/'weights.npz',[0]*5,root/'a')
            for bad in ([0]*4,[0,0,0,0,4097],[True,0,0,0,0],[0,0,0,0,'0}; int injected;']):
                with self.assertRaises(ValueError):assets.export_assets(root/'weights.npz',bad,root/'bad')
                self.assertFalse((root/'bad').exists())

    def test_corrupt_archive_and_symlink_inputs_or_outputs_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();model(root/'weights.npz');(root/'alias.npz').symlink_to(root/'weights.npz')
            with self.assertRaises(ValueError):assets.read_layers(root/'alias.npz')
            (root/'bad.npz').write_bytes(b'not a zip')
            with self.assertRaises(ValueError):assets.read_layers(root/'bad.npz')
            (root/'link').symlink_to(root,target_is_directory=True)
            with self.assertRaises(ValueError):assets.export_assets(root/'weights.npz',[0]*5,root/'link/new')
            self.assertFalse((root/'new').exists())


if __name__=='__main__':unittest.main()
