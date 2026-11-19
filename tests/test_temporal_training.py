"""学習/validation分離と元domainの同一性を守る。"""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import temporal_training as train
from arcade_ai_pet.channel_robustness import PROFILES
from test_temporal_model import fixture


class TemporalDispatchTests(unittest.TestCase):
    def fixture(self,root):
        domains={n:root/n for n in ('legacy','neural')};records=[]
        for name,path in domains.items():
            path.mkdir();row=dict(id=hashlib.sha256(name.encode()).hexdigest(),sha256=hashlib.sha256((name+'wave').encode()).hexdigest());records.append(row)
            (path/'manifest.json').write_text(json.dumps(dict(role='development',variants=1,records=[row])))
        mixed=root/'mixed';mixed.mkdir();(mixed/'manifest.json').write_text(json.dumps(dict(role='development',variants=1,records=records)))
        return domains,mixed

    def test_only_train_is_weighted_and_validation_keeps_sixteen_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains,mixed=self.fixture(root)
            def arrays(*args,**kwargs):
                self.assertEqual(kwargs['frontend'],'adaptive')
                y=np.tile(np.arange(7),8);x=np.zeros((len(y),260),dtype=np.float32)
                return dict(train=dict(x=x[:7],y=y[:7]),validation=dict(x=x,y=y,profiles=[p for p in PROFILES for _ in range(7)]))
            def fit(x,y,vx,vy,groups,**kwargs):
                self.assertEqual(len(y),21);self.assertEqual(len(vy),112);self.assertEqual(set(groups),set(range(16)))
                return dict(layers=fixture(),selected_epoch=1,checkpoint_candidates_evaluated=1)
            with patch.object(train,'make_arrays',side_effect=arrays),patch.object(train,'fit',side_effect=fit):
                result=train.train_candidate(domains,mixed,root/'candidate',epochs=5,neural_weight=2,device='cpu')
            self.assertEqual(result['training_device'],'cpu');self.assertEqual(result['classifier'],'temporal-cnn-v1');self.assertFalse(result['accepted_validation'])
            self.assertEqual(result['weights_bytes'],18280);self.assertFalse(result['final_evaluated'])
            self.assertEqual(len(result['domain_profiles']),16)

    def test_final_or_mismatched_mixed_domain_is_rejected_before_features(self):
        for mode in ('final','mixed','overlap'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();domains,mixed=self.fixture(root)
                path=domains['neural']/'manifest.json' if mode!='mixed' else mixed/'manifest.json';meta=json.loads(path.read_text())
                if mode=='final':meta['role']='final'
                elif mode=='mixed':meta['records'].reverse()
                else:meta['records']=json.loads((domains['legacy']/'manifest.json').read_text())['records']
                path.write_text(json.dumps(meta))
                with patch.object(train,'make_arrays',side_effect=AssertionError('no data read')):
                    with self.assertRaises(ValueError):train.train_candidate(domains,mixed,root/'candidate')
                self.assertFalse((root/'candidate').exists())


if __name__=='__main__':unittest.main()
