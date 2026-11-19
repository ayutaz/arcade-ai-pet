"""元発話/split/ライセンスを保った開発データの結合。"""
import hashlib,json,tempfile,unittest,wave
from pathlib import Path
import numpy as np
from arcade_ai_pet.mixed_corpus import merge_development
from arcade_ai_pet.features import features,split_for
from arcade_ai_pet.data import augment,read_pcm,trim_silence
from arcade_ai_pet.training import load_dataset
from arcade_ai_pet.native_evaluation import _examples


def fixture(path,namespace,role='development'):
    (path/'sources').mkdir(parents=True);records=[];arrays={s:dict(x=[],y=[]) for s in ('train','validation','test')}
    for index in range(60):
        source=f'{namespace}/speech/{index}';ident=hashlib.sha256(source.encode()).hexdigest();split=split_for(source)
        samples=np.rint(2000*np.sin(np.arange(3200)*(.03+int(ident[:8],16)/2**36))).astype('<i2');file=path/'sources'/(ident+'.wav')
        with wave.open(str(file),'wb') as stream:stream.setparams((1,2,16000,0,'NONE','not compressed'));stream.writeframes(samples.tobytes())
        records.append(dict(source=source,id=ident,split=split,label=index%6,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        arrays[split]['x'].append(features(augment(trim_silence(read_pcm(file)),int(ident[:8],16))))
        arrays[split]['y'].append(index%6)
    manifest=dict(schema=1,role=role,variants=1,feature_version='aligned-mfcc-v2',license=namespace+' fixture license',records=records)
    (path/'manifest.json').write_text(json.dumps(manifest))
    for split,values in arrays.items():np.savez_compressed(path/(split+'.npz'),x=np.array(values['x'],dtype=np.float32),y=np.array(values['y'],dtype=np.int64))
    return manifest


class MixedCorpusTests(unittest.TestCase):
    def test_merge_preserves_sources_splits_reconstruction_and_component_licenses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();a=fixture(root/'a','alpha');b=fixture(root/'b','beta')
            result=merge_development([root/'a',root/'b'],root/'mixed')
            self.assertEqual(result['speech_sources'],120);self.assertFalse(result['real_voice_tested'])
            actual=json.loads((root/'mixed/manifest.json').read_text())
            self.assertEqual(actual['records'],a['records']+b['records'])
            self.assertEqual({r['license'] for r in actual['components']},{a['license'],b['license']})
            self.assertEqual(len(list(_examples(root/'mixed'))),120)
            self.assertEqual(sum(len(v['y']) for v in load_dataset(root/'mixed').values()),120)

    def test_final_role_or_duplicate_ids_or_changed_wave_rejected_before_output(self):
        for mode in ('final','duplicate','changed'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();a=fixture(root/'a','alpha');fixture(root/'b','beta',role='final' if mode=='final' else 'development')
                inputs=[root/'a',root/'a'] if mode=='duplicate' else [root/'a',root/'b']
                if mode=='changed':(root/'a/sources'/(a['records'][0]['id']+'.wav')).write_bytes(b'changed')
                with self.assertRaises(ValueError):merge_development(inputs,root/'mixed')
                self.assertFalse((root/'mixed').exists())

    def test_speech_content_duplicate_across_different_ids_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();a=fixture(root/'a','alpha');b=fixture(root/'b','beta')
            raw=(root/'a/sources'/(a['records'][0]['id']+'.wav')).read_bytes()
            (root/'b/sources'/(b['records'][0]['id']+'.wav')).write_bytes(raw);b['records'][0]['sha256']=hashlib.sha256(raw).hexdigest()
            (root/'b/manifest.json').write_text(json.dumps(b))
            with self.assertRaises(ValueError):merge_development([root/'a',root/'b'],root/'mixed')
            self.assertFalse((root/'mixed').exists())

    def test_incompatible_variants_or_existing_output_are_not_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();fixture(root/'a','alpha');b=fixture(root/'b','beta');b['variants']=2
            (root/'b/manifest.json').write_text(json.dumps(b))
            with self.assertRaises(ValueError):merge_development([root/'a',root/'b'],root/'mixed')
            self.assertFalse((root/'mixed').exists())
            with self.assertRaises(ValueError):merge_development([root/'a'],root/'a')
            self.assertEqual(len(json.loads((root/'a/manifest.json').read_text())['records']),60)


if __name__=='__main__':unittest.main()
