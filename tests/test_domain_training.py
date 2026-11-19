"""新しい声の未達を旧データの量で隠さない学習/選定。"""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import domain_training as train
from arcade_ai_pet.channel_robustness import PROFILES
from arcade_ai_pet.frozen import read_layers
import test_frozen as fixture


def validation():
    labels=np.tile(np.arange(7),16);groups=np.repeat(np.arange(16),7)
    logits=np.eye(7,dtype=np.int64)[labels]*10
    return labels,groups,logits


class DomainTrainingTests(unittest.TestCase):
    def test_rank_rejects_one_bad_domain_profile_despite_many_old_examples(self):
        y,g,logits=validation();pred=logits.argmax(1)
        self.assertTrue(train.validation_rank(y,pred,g)[0])
        bad=pred.copy();bad[g==8]=0
        self.assertFalse(train.validation_rank(y,bad,g)[0])
        self.assertGreater(train.validation_rank(y,pred,g),train.validation_rank(y,bad,g))

    def test_threshold_search_uses_all_sixteen_groups_and_preserves_commands(self):
        y,g,logits=validation()
        indexes=np.where((g>=8)&(y==5))[0];logits[indexes]=0;logits[indexes,0]=2;logits[indexes,5]=1
        threshold,rows,accepted=train.select_threshold(y,logits,g)
        self.assertEqual(threshold,2);self.assertTrue(accepted);self.assertEqual(len(rows),16)
        self.assertTrue(all(row['macro_f1']==1 for row in rows))

    def test_missing_groups_and_out_of_bounds_inputs_are_rejected(self):
        y,g,logits=validation()
        for groups in (g%8,np.full(len(g),16),g.astype(float)):
            with self.assertRaises(ValueError):train.select_threshold(y,logits,groups)
        with self.assertRaises(ValueError):train.select_threshold(y,logits.astype(float),g)

    def test_training_weights_only_training_rows_and_refuses_final_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains={name:root/name for name in ('legacy','neural')};records=[]
            for name,path in domains.items():
                path.mkdir();row=dict(id=hashlib.sha256(name.encode()).hexdigest(),sha256=hashlib.sha256((name+'wave').encode()).hexdigest());records.append(row)
                (path/'manifest.json').write_text(json.dumps(dict(role='development',variants=1,records=[row])))
            mixed=root/'mixed';mixed.mkdir();(mixed/'manifest.json').write_text(json.dumps(dict(role='development',variants=1,records=records)))
            model=root/'fixture';fixture.checkpoint(model);layers,_=read_layers(model/'weights.npz')
            def arrays(path,*args,**kwargs):
                code=1 if path==domains['legacy'] else 2;y=np.arange(7);x=np.zeros((7,260),dtype=np.float32);x[np.arange(7),y]=4;x[:,259]=code
                vy=np.tile(y,8);vx=np.tile(x,(8,1));vx[:,259]=code+2
                return dict(train=dict(x=x,y=y),validation=dict(x=vx,y=vy,profiles=[p for p in PROFILES for _ in range(7)]))
            with patch.object(train,'make_arrays',side_effect=arrays),patch.object(train,'fit',return_value=dict(layers=layers,selected_epoch=1)) as fit:
                result=train.train_candidate(domains,mixed,root/'candidate',epochs=5,hidden=7,neural_weight=4)
            args=fit.call_args.args
            self.assertEqual(len(args[0]),35);self.assertEqual(set(args[0][:,259]),{1,2})
            self.assertEqual(set(args[2][:,259]),{3,4});self.assertEqual(set(args[4]),set(range(16)))
            self.assertTrue(result['accepted_all_domains']);self.assertFalse(result['final_evaluated'])
            p=domains['neural']/'manifest.json';m=json.loads(p.read_text());m['role']='final';p.write_text(json.dumps(m))
            with patch.object(train,'make_arrays',side_effect=AssertionError('no final features')):
                with self.assertRaises(ValueError):train.train_candidate(domains,mixed,root/'rejected')
            self.assertFalse((root/'rejected').exists())

    def test_checkpoint_selection_compares_calibrated_gates_instead_of_raw_rank(self):
        y=np.tile(np.array([0,1,2,3,4,5,6,5,5,5]),16);g=np.repeat(np.arange(16),10)
        a=np.eye(7,dtype=np.int64)[y]*10;b=a.copy()
        a[0]=0;a[0,1]=10
        indexes=np.where(y==5)[0];b[indexes]=0;b[indexes,0]=2;b[indexes,5]=1
        self.assertGreater(train.validation_rank(y,a.argmax(1),g),train.validation_rank(y,b.argmax(1),g))
        with patch.object(train,'scores',side_effect=[a,b]):
            selected=train.select_checkpoint([dict(layers='raw-best',epoch=10),dict(layers='calibrated-best',epoch=20)],np.zeros((len(y),260)),y,g)
        self.assertEqual(selected['selected_epoch'],20);self.assertEqual(selected['threshold_q10'],2)
        self.assertTrue(selected['accepted_all_domains'])

    def test_shortlist_is_bounded_and_keeps_best_of_each_training_period(self):
        checkpoints=[dict(layers=[],epoch=i,rank=(False,i/201,i/201)) for i in range(1,202)]
        short=train.shortlist(checkpoints)
        self.assertLessEqual(len(short),16);self.assertEqual(len({r['epoch'] for r in short}),len(short))
        self.assertIn(201,{r['epoch'] for r in short})
        self.assertEqual({min(7,(r['epoch']-1)*8//201) for r in short},set(range(8)))

    def test_context_mode_uses_context_builder_and_refuses_legacy_freeze(self):
        from arcade_ai_pet import channel_freeze
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();domains={name:root/name for name in ('legacy','neural')};records=[]
            for name,path in domains.items():
                path.mkdir();row=dict(id=hashlib.sha256(name.encode()).hexdigest(),sha256=hashlib.sha256((name+'wave').encode()).hexdigest());records.append(row)
                (path/'manifest.json').write_text(json.dumps(dict(role='development',variants=1,records=[row])))
            mixed=root/'mixed';mixed.mkdir();(mixed/'manifest.json').write_text(json.dumps(dict(role='development',variants=1,records=records)))
            model=root/'fixture';fixture.checkpoint(model);layers,_=read_layers(model/'weights.npz')
            def arrays(*args,**kwargs):
                y=np.tile(np.arange(7),8);x=np.zeros((56,260),dtype=np.float32);x[np.arange(56),y]=4
                return dict(train=dict(x=x,y=y),validation=dict(x=x,y=y,profiles=[p for p in PROFILES for _ in range(7)]))
            with patch.object(train,'make_arrays',side_effect=AssertionError('no adaptive arrays')),patch.object(train,'load_context_arrays',side_effect=arrays,create=True),patch.object(train,'fit',return_value=dict(layers=layers,selected_epoch=1)):
                result=train.train_candidate(domains,mixed,root/'candidate',epochs=5,hidden=7,frontend='context')
            self.assertEqual(result['frontend_version'],'context');self.assertIn('context_frontend.py',result['feature_sha256'])
            with patch.object(channel_freeze,'make_arrays',side_effect=AssertionError('no legacy feature export')):
                with self.assertRaises(ValueError):channel_freeze.freeze_channel(root/'candidate',mixed,root/'legacy-frozen')
            self.assertFalse((root/'legacy-frozen').exists())


if __name__=='__main__':unittest.main()
