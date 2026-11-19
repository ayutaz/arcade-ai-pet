"""新形式のfreezeは元音声/C/全groupを再検査し、成功flagを信用しない。"""
import hashlib,json,tempfile,unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch,Mock
import numpy as np
from arcade_ai_pet import decision_freeze as freeze
from arcade_ai_pet.decision_assets import read_layers
from arcade_ai_pet.training import scores
from arcade_ai_pet.class_rejection import decisions
from test_neural_corpus import plan as source_plan


def plan():
    p=source_plan()
    p['conditions'] += [dict(id=f'final-0{i}',role='final',caption=f'最終条件の声{i}。') for i in range(2,5)]
    return p


def fixture(root):
    weights=root/'weights.npz';layers=[dict(weight=np.pad(np.eye(7,dtype=np.int8),((0,0),(0,254))),bias=np.zeros(7,dtype=np.int32),shift=0),
                                     dict(weight=np.eye(7,dtype=np.int8),bias=np.zeros(7,dtype=np.int32),shift=0)]
    np.savez(weights,**{f'{k}{i}':r[k] for i,r in enumerate(layers) for k in ('weight','bias','shift')})
    domains={};examples={}
    for name in ('legacy','neural'):
        path=root/name;path.mkdir();records=[];data=[]
        for label in range(7):
            source=f'{name}/{label}';ident=hashlib.sha256(source.encode()).hexdigest()
            records.append(dict(source=source,id=ident,split='validation',label=label,sha256=hashlib.sha256(('wave/'+source).encode()).hexdigest()))
            data.append((np.full(32000,label+1,dtype=np.int16),label,None))
        (path/'manifest.json').write_text(json.dumps(dict(variants=1,role='development',records=records)))
        domains[name]=path;examples[path]=data
    def feature(pcm,*args):
        x=np.zeros(261,dtype=np.float32);x[int(pcm[0])-1]=4;return x
    class Native:
        scratch_bytes=5680
        def __init__(self,assets):
            self.layers,_=read_layers(Path(assets)/'weights.npz')
            self.table=np.asarray(json.loads((Path(assets)/'decision-metadata.json').read_text())['thresholds_q10'],dtype=np.int32)
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def extract(self,pcm,*,banked=False):return feature(pcm)
        def prepare(self,pcm):return pcm
        def predict(self,x):
            logit=scores(x[None,:],self.layers)[0].astype(np.int32)
            return int(decisions(logit[None,:],self.table)[0]),logit
    return weights,domains,examples,feature,Native


def patched(root):
    weights,domains,examples,feature,native=fixture(root);stack=ExitStack()
    stack.enter_context(patch.object(freeze,'_examples',side_effect=lambda path:iter(examples[path])))
    stack.enter_context(patch.object(freeze,'_features',side_effect=feature))
    stack.enter_context(patch.object(freeze,'simulate',side_effect=lambda pcm,*args:pcm))
    stack.enter_context(patch.object(freeze,'DecisionNativeModel',native))
    return stack,weights,domains,examples


class DecisionFreezeTests(unittest.TestCase):
    def test_freeze_reconstructs_all_domains_and_native_paths_before_fresh_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();stack,w,domains,_=patched(root)
            with stack:r=freeze.freeze_candidate(w,[10]*5,domains,plan(),root/'frozen')
            self.assertEqual(r['architecture'],[261,7,7]);self.assertEqual(r['thresholds_q10'],[10]*5)
            self.assertEqual(len(r['domain_profiles']),16);self.assertEqual(len(r['pooled_profiles']),8)
            self.assertEqual(r['conformance']['rows'],112);self.assertTrue(r['source_sha256'])
            self.assertFalse(r['final_evaluated']);self.assertFalse(r['device_tested'])
            meta,_=freeze.verify(root/'frozen');self.assertEqual(meta,r)
            with self.assertRaises(ValueError):freeze.freeze_candidate(w,[10]*5,domains,plan(),root/'frozen')

    def test_bad_group_quality_cannot_be_hidden_by_success_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();stack,w,domains,examples=patched(root)
            examples[domains['neural']][0]=(np.full(32000,2,dtype=np.int16),0,None)
            with stack:
                with self.assertRaises(ValueError):freeze.freeze_candidate(w,[10]*5,domains,plan(),root/'frozen')
            self.assertFalse((root/'frozen').exists())

    def test_final_role_overlapping_sources_and_reserved_conditions_refuse_before_pcm(self):
        for mode in ('role','source','wave','condition'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();w,domains,_,_,_=fixture(root)
                p=domains['neural']/'manifest.json';m=json.loads(p.read_text())
                legacy=json.loads((domains['legacy']/'manifest.json').read_text())
                if mode=='role':m['role']='final'
                elif mode=='source':m['records'][0]['id']=legacy['records'][0]['id']
                elif mode=='wave':m['records'][0]['sha256']=legacy['records'][0]['sha256']
                else:m['records'][0]['condition_id']='final-01'
                p.write_text(json.dumps(m))
                with patch.object(freeze,'_examples',side_effect=AssertionError('PCM reached')):
                    with self.assertRaises(ValueError):freeze.freeze_candidate(w,[10]*5,domains,plan(),root/'frozen')
                self.assertFalse((root/'frozen').exists())

    def test_modified_code_weights_assets_or_reserved_plan_refuse_verification(self):
        for mode in ('code','weights','header','plan','table'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();stack,w,domains,_=patched(root)
                with stack:freeze.freeze_candidate(w,[10]*5,domains,plan(),root/'frozen')
                frozen=root/'frozen';p=frozen/'decision-frozen.json';meta=json.loads(p.read_text())
                if mode=='code':meta['source_sha256']['arcade_ai_pet/training.py']='0'*64;p.write_text(json.dumps(meta))
                elif mode=='weights':(frozen/'weights.npz').write_bytes(b'changed')
                elif mode=='header':(frozen/'decision_model.h').write_text('changed')
                elif mode=='table':meta['thresholds_q10'][0]=20;p.write_text(json.dumps(meta))
                else:(frozen/'final-plan.json').write_text(json.dumps(source_plan()))
                with self.assertRaises(ValueError):freeze.verify(frozen)


if __name__=='__main__':unittest.main()
