"""durationの明示選択・train限定の比重・旧export拒否の契約。"""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import domain_training as train
from arcade_ai_pet.channel_robustness import PROFILES
from arcade_ai_pet.channel_freeze import freeze_channel
from arcade_ai_pet.frozen import read_layers


class DurationDispatchTests(unittest.TestCase):
    def test_explicit_duration_mode_uses_261_features_and_preserves_validation(self):
        self.trial('duration')

    def test_explicit_context_duration_mode_keeps_261_inputs_and_rejects_old_export(self):
        self.trial('context-duration')

    def test_explicit_cmn_mode_keeps_261_inputs_and_records_the_new_feature_contract(self):
        self.trial('context-duration-cmn')

    def trial(self,frontend):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains={n:root/n for n in ('legacy','neural')};records=[]
            for name,path in domains.items():
                path.mkdir();row=dict(id=hashlib.sha256(name.encode()).hexdigest(),sha256=hashlib.sha256((name+'wave').encode()).hexdigest());records.append(row)
                (path/'manifest.json').write_text(json.dumps(dict(role='development',variants=1,records=[row])))
            mixed=root/'mixed';mixed.mkdir();(mixed/'manifest.json').write_text(json.dumps(dict(role='development',variants=1,records=records)))
            weight=np.zeros((7,261),dtype=np.int8);weight[np.arange(7),np.arange(7)]=64
            layers=[dict(weight=weight,bias=np.zeros(7,dtype=np.int32),shift=6),dict(weight=np.eye(7,dtype=np.int8)*64,bias=np.zeros(7,dtype=np.int32),shift=6)]
            def arrays(*args,**kwargs):
                self.assertEqual(kwargs['frontend'],frontend)
                y=np.arange(7);x=np.zeros((7,261),dtype=np.float32);x[np.arange(7),y]=4;x[:,-1]=2
                return dict(train=dict(x=x,y=y),validation=dict(x=np.tile(x,(8,1)),y=np.tile(y,8),profiles=[p for p in PROFILES for _ in range(7)]))
            builder='load_cmn_arrays' if frontend=='context-duration-cmn' else 'load_duration_arrays'
            with patch.object(train,'make_arrays',side_effect=AssertionError('old builder used')),patch.object(train,builder,side_effect=arrays,create=True),patch.object(train,'fit',return_value=dict(layers=layers,selected_epoch=1)) as fit:
                result=train.train_candidate(domains,mixed,root/'candidate',epochs=5,hidden=7,neural_weight=2,frontend=frontend)
            self.assertEqual(fit.call_args.kwargs['feature_count'],261)
            self.assertEqual(len(fit.call_args.args[1]),21);self.assertEqual(len(fit.call_args.args[3]),112)
            self.assertEqual(set(fit.call_args.args[4]),set(range(16)))
            self.assertEqual(result['frontend_version'],frontend);self.assertTrue(result['accepted_validation'])
            self.assertIn('duration_frontend.py',result['feature_sha256']);self.assertIn('duration_arrays.py',result['feature_sha256'])
            if frontend in ('context-duration','context-duration-cmn'):self.assertIn('context_frontend.py',result['feature_sha256'])
            if frontend=='context-duration-cmn':
                self.assertIn('cmn_frontend.py',result['feature_sha256']);self.assertIn('cmn_arrays.py',result['feature_sha256'])
            with self.assertRaises(ValueError):read_layers(root/'candidate/weights.npz')
            with self.assertRaises(ValueError):freeze_channel(root/'candidate',mixed,root/'old-frozen')
            self.assertFalse((root/'old-frozen').exists())


if __name__=='__main__':unittest.main()
