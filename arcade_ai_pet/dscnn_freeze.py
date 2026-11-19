"""Freeze the complete logmel CNN/runtime and the unchanged caption holdout."""
from pathlib import Path
import hashlib,json,shutil,tempfile
import numpy as np
from . import dscnn_assets as assets
from .dscnn_assets import DscnnNativeAssets as NativeModel
from .dscnn_model import scores,SHAPES
from .dscnn_candidate import SOURCES as MODEL_SOURCES
from .dscnn_native import SOURCES as CORE_SOURCES
from .logmel_native import SOURCES as FRONTEND_SOURCES
from .logmel_frontend import features
from .channel_robustness import PROFILES,simulate,condition,measure
from .channel_training import level
from .native_evaluation import _examples
from .caption_holdout import select_holdout
from .caption_training import _input_hashes
from .neural_corpus import validate_plan
from .class_rejection import decisions

ROOT=Path(__file__).resolve().parents[1]
SOURCES=tuple(dict.fromkeys((*MODEL_SOURCES,*CORE_SOURCES,*FRONTEND_SOURCES,
    'arcade_ai_pet/dscnn_assets.py','arcade_ai_pet/dscnn_freeze.py','arcade_ai_pet/decision_assets.py',
    'arcade_ai_pet/decision_freeze.py','arcade_ai_pet/decision_final.py','arcade_ai_pet/native.py',
    'arcade_ai_pet/neural_corpus.py','arcade_ai_pet/generated_audio.py','arcade_ai_pet/host_teacher.py')))


def source_hashes():return {name:assets.digest(ROOT/name) for name in SOURCES}
def _features(pcm,profile,seed):return features(level(condition(simulate(pcm,profile,seed),'hp160')))
def _accepted(rows):return bool(rows) and all(r['macro_f1']>=.9 and r['false_activation_rate']<=.05 and r['command_recall']>=.8 for r in rows)


def recompute(paths,directory,layers,table,*,validation_only,selection=None):
    if validation_only:
        if set(paths)!={'legacy','neural'} or not isinstance(selection,dict) or select_holdout(paths['neural'],seed=selection.get('seed'))!=selection:
            raise ValueError('exact original caption holdout required')
        names=['legacy','held-neural','familiar-neural']
    else:names=list(paths)
    labels=[];predicted=[];groups=[]
    stats=dict(rows=0,exact_integer_logits=0,banked_features_equal=0,final_decisions_equal=0,q5_features_equal=0,maximum_feature_error=0.)
    with NativeModel(directory) as engine:
        for name,dataset in paths.items():
            manifest=json.loads((dataset/'manifest.json').read_text());expanded=[r for r in manifest['records'] for _ in range(manifest['variants'] if 'sha256' in r else 1)]
            for (pcm,label,_),row in zip(_examples(dataset),expanded,strict=True):
                if label!=row['label']:raise ValueError('reconstructed label changed')
                domain=name
                if validation_only:
                    if name=='legacy':domain='legacy' if row['split']=='validation' else None
                    elif row['split']=='test':domain=None
                    elif row['split']=='validation':domain='familiar-neural'
                    elif row.get('condition_id') in selection['held_condition_ids'] or row['id'] in selection['held_control_ids']:domain='held-neural'
                    else:domain=None
                    if domain is None:continue
                seed=int(row['id'][:16],16)
                for index,profile in enumerate(PROFILES):
                    expected=_features(pcm,profile,seed);raw=scores(expected[None,:],layers)[0]
                    wanted=int(decisions(raw[None,:],table)[0]);changed=simulate(pcm,profile,seed)
                    actual=engine.extract(changed);banked=engine.extract(changed,banked=True);result,logits=engine.predict(actual)
                    stats['rows']+=1;stats['exact_integer_logits']+=int(np.array_equal(raw,logits))
                    stats['banked_features_equal']+=int(np.array_equal(actual,banked));stats['final_decisions_equal']+=int(result==wanted)
                    stats['q5_features_equal']+=int(np.array_equal(np.rint(expected*32),np.rint(actual*32)))
                    stats['maximum_feature_error']=max(stats['maximum_feature_error'],float(abs(expected-actual).max()))
                    labels.append(label);predicted.append(result);groups.append(names.index(domain)*8+index)
        stats['scratch_bytes']=engine.scratch_bytes
    y,p,g=np.asarray(labels),np.asarray(predicted),np.asarray(groups)
    if not len(y) or set(g.tolist())!=set(range(len(names)*8)):raise ValueError('all named domain/profile groups required')
    rows=[dict(domain=name,profile=profile,**measure(y[g==i*8+j],p[g==i*8+j])) for i,name in enumerate(names) for j,profile in enumerate(PROFILES)]
    pooled=[dict(profile=profile,**measure(y[g%8==j],p[g%8==j])) for j,profile in enumerate(PROFILES)]
    stats['matched']=bool(all(stats[k]==stats['rows'] for k in ('exact_integer_logits','banked_features_equal','final_decisions_equal','q5_features_equal')) and stats['maximum_feature_error']<=2e-6 and stats['scratch_bytes']<=32768)
    return rows,pooled,stats


def freeze_candidate(weights,table,domains,plan,output,*,holdout_seed):
    from .decision_freeze import _domains
    output=assets.real_path(output)
    if output.exists() or type(holdout_seed) is not int or not 0<=holdout_seed<2**31:raise ValueError('fresh output and explicit holdout seed required')
    plan_hash=validate_plan(plan);final_ids=[r['id'] for r in plan['conditions'] if r['role']=='final']
    if len(final_ids)!=4:raise ValueError('four unused final conditions required')
    paths,ids,waves,manifests=_domains(domains,set(final_ids))
    if set(paths)!={'legacy','neural'}:raise ValueError('original legacy/neural domains required')
    selection=select_holdout(paths['neural'],seed=holdout_seed);before=_input_hashes(paths);code=source_hashes()
    weights=assets.real_path(weights);weight_hash=assets.digest(weights);layers,size=assets.read_layers(weights);table=assets.thresholds(table)
    with tempfile.TemporaryDirectory() as tmp:
        numeric=Path(tmp).resolve()/'assets';asset_meta=assets.export_assets(weights,table,numeric)
        rows,pooled,conformance=recompute(paths,numeric,layers,np.array(table,np.int32),validation_only=True,selection=selection)
        if not _accepted(rows) or not _accepted(pooled) or not conformance['matched']:raise ValueError('development quality/full native gates failed')
        if before!=_input_hashes(paths) or code!=source_hashes() or weight_hash!=assets.digest(weights):raise ValueError('freeze inputs/source changed')
        result=dict(schema=1,runtime=assets.RUNTIME,input_contract=assets.CONTRACT.copy(),architecture=[list(s) for s in SHAPES],
            thresholds_q10=table,weights_sha256=weight_hash,weights_bytes=size,numeric_bytes=size+20,source_sha256=code,
            development_manifest_sha256=manifests,development_input_sha256=before,selection=selection,source_ids=ids,wave_hashes=waves,
            plan_sha256=plan_hash,final_condition_ids=final_ids,domain_profiles=rows,pooled_profiles=pooled,conformance=conformance,
            asset_metadata_sha256=assets.digest(numeric/'decision-metadata.json'),header_sha256=asset_meta['header_sha256'],
            final_evaluated=False,device_tested=False,real_voice_tested=False,license='Model/data conditions are separate; see provenance and data-license.md')
        output.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(numeric,output)
    (output/'final-plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    (output/'decision-frozen.json').write_text(json.dumps(result,indent=2)+'\n');return result


def verify(directory):
    directory=assets.real_path(directory);meta=json.loads(assets.real_path(directory/'decision-frozen.json').read_text())
    table=assets.thresholds(meta['thresholds_q10'])
    if (meta.get('schema')!=1 or meta.get('runtime')!=assets.RUNTIME or meta.get('input_contract')!=assets.CONTRACT or
        meta.get('source_sha256')!=source_hashes() or meta.get('architecture')!=[list(s) for s in SHAPES]):raise ValueError('frozen source/runtime/architecture changed')
    layers,size=assets.read_layers(directory/'weights.npz')
    if (assets.digest(directory/'weights.npz')!=meta['weights_sha256'] or meta['weights_bytes']!=size or meta['numeric_bytes']!=size+20):raise ValueError('frozen numeric model changed')
    plan=json.loads(assets.real_path(directory/'final-plan.json').read_text())
    if (validate_plan(plan)!=meta['plan_sha256'] or [r['id'] for r in plan['conditions'] if r['role']=='final']!=meta['final_condition_ids'] or len(meta['final_condition_ids'])!=4):raise ValueError('frozen final plan changed')
    if assets.digest(assets.real_path(directory/'decision-metadata.json'))!=meta['asset_metadata_sha256']:raise ValueError('frozen numeric metadata changed')
    numeric=json.loads((directory/'decision-metadata.json').read_text())
    if (numeric['thresholds_q10']!=table or numeric['weights_sha256']!=meta['weights_sha256'] or numeric['runtime']!=assets.RUNTIME or numeric['input_contract']!=assets.CONTRACT):raise ValueError('frozen numeric binding changed')
    for name,text in assets.render(layers,table).items():
        expected=hashlib.sha256(text.encode()).hexdigest()
        if assets.digest(assets.real_path(directory/name))!=expected or meta['header_sha256'].get(name)!=expected:raise ValueError('frozen header changed')
    return meta,layers
