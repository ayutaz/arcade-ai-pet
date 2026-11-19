"""caption単位の保留と全加工例の排除を、既存archiveを変えずに検査する。"""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import caption_holdout as holdout
from arcade_ai_pet.features import split_for
from arcade_ai_pet.channel_robustness import PROFILES
from arcade_ai_pet.cmn_arrays import make_arrays as previous
from test_mixed_corpus import fixture as speech_fixture

SEED = 2026111904
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source_row(name, label, split, condition=None):
    for nonce in range(10000):
        source=f'{name}/{nonce}'
        if split_for(source)==split:break
    ident=hashlib.sha256(source.encode()).hexdigest()
    row=dict(id=ident,source=source,label=label,split=split)
    if condition is not None:
        row.update(condition_id=condition,sha256=hashlib.sha256(('wave/'+source).encode()).hexdigest())
    return row


def fixture(root):
    records=[source_row(f'group-{g}-label-{label}',label,'train',f'caption-{g}') for g in range(5) for label in range(6)]
    records += [source_row('familiar-train',0,'train','familiar'),source_row('familiar-val',1,'validation','familiar'),source_row('familiar-test',2,'test','familiar')]
    for index in range(200):
        source=f'fixture-noise/{index}';ident=hashlib.sha256(source.encode()).hexdigest()
        records.append(dict(id=ident,source=source,label=6,split=split_for(source)))
    manifest=dict(role='development',feature_version='aligned-mfcc-v2',variants=2,records=records)
    (root/'manifest.json').write_text(json.dumps(manifest))
    for split in ('train','validation','test'):(root/(split+'.npz')).write_bytes(('original '+split).encode())
    expanded=[(row,v) for row in records for v in range(2 if 'sha256' in row else 1)]
    data=[(np.full(32000,index+1,dtype=np.int16),row['label'],None) for index,(row,v) in enumerate(expanded)]
    return manifest,expanded,data


def fake_features(pcm):
    return np.full(261,int(pcm[0])/1000,dtype=np.float32)


def patches(data):
    stack=ExitStack()
    stack.enter_context(patch.object(holdout,'_examples',return_value=iter(data)))
    stack.enter_context(patch.object(holdout,'simulate',side_effect=lambda pcm,*args:pcm))
    stack.enter_context(patch.object(holdout,'condition',side_effect=lambda pcm,*args:pcm))
    stack.enter_context(patch.object(holdout,'level',side_effect=lambda pcm:pcm))
    stack.enter_context(patch.object(holdout,'features',side_effect=fake_features))
    return stack


class CaptionHoldoutTests(unittest.TestCase):
    def test_selection_is_deterministic_and_uses_only_original_train_groups_and_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();manifest,_,_=fixture(root)
            selected=holdout.select_holdout(root,seed=SEED)
            expected=sorted([f'caption-{i}' for i in range(5)],key=lambda value:hashlib.sha256(f'{SEED}/condition/{value}'.encode()).hexdigest())[:4]
            controls=sorted([r['id'] for r in manifest['records'] if r['label']==6 and r['split']=='train'],key=lambda value:hashlib.sha256(f'{SEED}/control/{value}'.encode()).hexdigest())[:20]
            self.assertEqual(selected,dict(seed=SEED,manifest_sha256=sha(root/'manifest.json'),held_condition_ids=expected,held_control_ids=controls))
            self.assertEqual(holdout.select_holdout(root,seed=SEED),selected)
            self.assertNotIn('familiar',selected['held_condition_ids'])

    def test_all_variants_and_train_replicas_are_disjoint_from_held_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();manifest,expanded,data=fixture(root);selected=holdout.select_holdout(root,seed=SEED)
            before={p.name:sha(p) for p in root.iterdir()}
            with patches(data):arrays=holdout.make_arrays(root,selection=selected,noise_replicas=5,hum_replicas=3)
            self.assertEqual(set(arrays),{'train','validation','held_validation'})
            ids={s:set(a['source_ids']) for s,a in arrays.items()}
            self.assertFalse(ids['train'] & ids['held_validation'])
            self.assertFalse(ids['validation'] & ids['held_validation'])
            self.assertFalse(ids['train'] & ids['validation'])
            expected_held={r['id'] for r in manifest['records'] if r.get('condition_id') in selected['held_condition_ids'] or r['id'] in selected['held_control_ids']}
            self.assertEqual(ids['held_validation'],expected_held)
            self.assertEqual(ids['validation'],{r['id'] for r in manifest['records'] if r['split']=='validation'})
            self.assertFalse({r['id'] for r in manifest['records'] if r['split']=='test'} & set.union(*ids.values()))
            for row,variant in expanded:
                if row['split']=='test':continue
                split='validation' if row['split']=='validation' else 'held_validation' if row['id'] in expected_held else 'train'
                a=arrays[split];matches=[i for i,(ident,v) in enumerate(zip(a['source_ids'],a['variants'],strict=True)) if ident==row['id'] and v==variant]
                self.assertEqual(len(matches),14 if split=='train' else 8)
                self.assertEqual([a['profiles'][i] for i in matches].count('noise'),5 if split=='train' else 1)
                self.assertTrue(all(int(a['y'][i])==row['label'] for i in matches))
            self.assertEqual(before,{p.name:sha(p) for p in root.iterdir()})

    def test_no_test_features_and_training_noise_seed_matches_the_existing_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();_,expanded,data=fixture(root);selected=holdout.select_holdout(root,seed=SEED)
            forbidden={i+1 for i,(row,_) in enumerate(expanded) if row['split']=='test'}
            def checked(pcm):
                if int(pcm[0]) in forbidden:raise AssertionError('test features reached')
                return fake_features(pcm)
            with patches(data),patch.object(holdout,'features',side_effect=checked),patch.object(holdout,'simulate',side_effect=lambda pcm,*args:pcm) as simulate:
                arrays=holdout.make_arrays(root,selection=selected,noise_replicas=2,hum_replicas=1)
            row,variant=next((r,v) for r,v in expanded if r['split']=='train' and r.get('condition_id') not in selected['held_condition_ids'] and r['id'] not in selected['held_control_ids'])
            index=expanded.index((row,variant));pcm=data[index][0]
            observed=[seed for values,profile,seed in (call.args for call in simulate.call_args_list) if values is pcm and profile=='noise']
            expected=[int(row['id'][:16],16),int.from_bytes(hashlib.sha256(row['id'].encode()+pcm.tobytes()+bytes([1])).digest()[:8],'little')]
            self.assertEqual(observed,expected)
            self.assertEqual(arrays['held_validation']['x'].shape[1],261)

    def test_final_or_changed_identity_or_bad_selection_is_refused_before_pcm(self):
        for mode in ('final','manifest','condition','controls','duplicate-id','mixed-group','seed'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();m,_,_=fixture(root);selected=holdout.select_holdout(root,seed=SEED)
                if mode=='final':m['role']='final'
                elif mode=='manifest':selected['manifest_sha256']='0'*64
                elif mode=='condition':selected['held_condition_ids'][0]='familiar'
                elif mode=='controls':selected['held_control_ids'][0]=selected['held_control_ids'][1]
                elif mode=='duplicate-id':m['records'][1]=m['records'][0].copy()
                elif mode=='mixed-group':m['records'][0]['split']='validation'
                else:selected['seed']=True
                (root/'manifest.json').write_text(json.dumps(m))
                with patch.object(holdout,'_examples',side_effect=AssertionError('PCM reached')):
                    with self.assertRaises(ValueError):holdout.make_arrays(root,selection=selected)

    def test_manifest_replacement_between_parse_and_selection_cannot_reuse_old_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();manifest,_,data=fixture(root);path=root/'manifest.json';original=path.read_bytes()
            manifest['records'][0]['condition_id']='another-caption'
            path.write_text(json.dumps(manifest));replacement=path.read_bytes();selected=holdout.select_holdout(root,seed=SEED)
            path.write_bytes(original)
            def replace(*args,**kwargs):
                path.write_bytes(replacement)
                return selected
            with patches(data),patch.object(holdout,'select_holdout',side_effect=replace):
                with self.assertRaises(ValueError):holdout.make_arrays(root,selection=selected)

    def test_reconstruction_count_or_label_mismatch_cannot_silently_change_membership(self):
        for mode in ('short','label'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();_,_,data=fixture(root);selected=holdout.select_holdout(root,seed=SEED)
                if mode=='short':data=data[:-1]
                else:data[0]=(data[0][0],6,None)
                with patches(data):
                    with self.assertRaises(ValueError):holdout.make_arrays(root,selection=selected)

    def test_archive_mutation_during_reconstruction_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();_,_,data=fixture(root);selected=holdout.select_holdout(root,seed=SEED)
            def mutate():
                yield from data
                (root/'validation.npz').write_bytes(b'external mutation')
            with patches(data),patch.object(holdout,'_examples',return_value=mutate()):
                with self.assertRaises(ValueError):holdout.make_arrays(root,selection=selected)

    def test_real_original_validation_features_are_identical_and_archives_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();m=speech_fixture(root/'data','caption-real');dataset=root/'data'
            training=0
            for row in m['records']:
                row['condition_id']=f'caption-{training%5}' if row['split']=='train' else 'familiar'
                if row['split']=='train':training+=1
            # Reconstruct the existing noise recipe and append twenty train controls.
            from arcade_ai_pet.features import features
            original=np.load(dataset/'train.npz',allow_pickle=False);x=list(original['x']);y=list(original['y']);original.close()
            added=0
            for index in range(200):
                source=f'caption-noise/{index}'
                if split_for(source)!='train':continue
                ident=hashlib.sha256(source.encode()).hexdigest();rng=np.random.default_rng(int.from_bytes(hashlib.sha256(source.encode()).digest()[:8],'little'))
                pcm=np.clip(rng.normal(0,0 if index%4==0 else rng.uniform(1,200),32000),-32767,32767).astype(np.int16)
                m['records'].append(dict(source=source,id=ident,label=6,split='train'));x.append(features(pcm));y.append(6);added+=1
                if added==20:break
            np.savez_compressed(dataset/'train.npz',x=np.asarray(x,dtype=np.float32),y=np.asarray(y,dtype=np.int64))
            (dataset/'manifest.json').write_text(json.dumps(m))
            before={p.name:sha(p) for p in dataset.glob('*.npz')};selected=holdout.select_holdout(dataset,seed=SEED)
            expected=previous(dataset,'hp160',noise_replicas=1,hum_replicas=1)['validation']
            actual=holdout.make_arrays(dataset,selection=selected,noise_replicas=1,hum_replicas=1)['validation']
            np.testing.assert_array_equal(actual['x'],expected['x']);np.testing.assert_array_equal(actual['y'],expected['y']);self.assertEqual(actual['profiles'],expected['profiles'])
            self.assertEqual(before,{p.name:sha(p) for p in dataset.glob('*.npz')})


if __name__=='__main__':unittest.main()
