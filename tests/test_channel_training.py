import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import channel_training as training
from arcade_ai_pet.channel_robustness import PROFILES


class ChannelTrainingTests(unittest.TestCase):
    def test_level_preserves_zero_and_uses_full_int16_range_without_wrapping(self):
        zero=np.zeros(32000,dtype=np.int16)
        np.testing.assert_array_equal(training.level(zero),zero)
        pcm=zero.copy();pcm[100]=-32768;pcm[101]=32767
        saved=pcm.copy();result=training.level(pcm)
        self.assertEqual(result.dtype,np.int16);self.assertEqual(result.min(),-4096)
        self.assertLessEqual(result.max(),4096);np.testing.assert_array_equal(pcm,saved)
        pcm[100]=-2;pcm[101]=2
        result=training.level(pcm);self.assertEqual(result[100],-4096);self.assertEqual(result[101],4096)
        for bad in (np.zeros(32000),np.zeros(31999,dtype=np.int16)):
            with self.assertRaises(ValueError):training.level(bad)

    def test_training_arrays_keep_source_splits_and_exclude_original_test_from_fitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);records=[];examples=[]
            for label,split in enumerate(('train','validation','test')):
                source='unit/'+split
                records.append(dict(source=source,id=hashlib.sha256(source.encode()).hexdigest(),split=split,label=label,sha256=str(label)*64))
                pcm=np.zeros(32000,dtype=np.int16);pcm[4000:12000]=(1000*np.sin(np.arange(8000)*.1)).astype(np.int16)
                examples.append((pcm,label,np.zeros(260,dtype=np.float32)))
            (root/'manifest.json').write_text(json.dumps(dict(variants=1,records=records)))
            with patch.object(training,'_examples',return_value=iter(examples)):
                data=training.make_arrays(root,'hp80')
            self.assertEqual(set(data),{'train','validation'})
            for split,label in [('train',0),('validation',1)]:
                self.assertEqual(data[split]['x'].shape,(len(PROFILES),260))
                self.assertEqual(data[split]['y'].tolist(),[label]*len(PROFILES))
                self.assertEqual(data[split]['profiles'],list(PROFILES))
                self.assertTrue(np.isfinite(data[split]['x']).all())
                self.assertGreater(np.linalg.norm(data[split]['x'][1]),1)

    def test_unknown_conditioner_cannot_create_training_data(self):
        with self.assertRaises(ValueError):training.make_arrays(Path('absent'),'unreviewed')

    def test_feature_variant_is_explicit_and_unknown_variants_fail_before_io(self):
        with self.assertRaises(ValueError):training.make_arrays(Path('absent'),'hp80',frontend='unknown')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            records=[dict(id='a'*64,split=s,sha256=str(i)*64) for i,s in enumerate(('train','validation'))]
            (root/'manifest.json').write_text(json.dumps(dict(variants=1,records=records)))
            pcm=np.zeros(32000,dtype=np.int16)
            examples=[(pcm,6,np.zeros(260,dtype=np.float32)) for _ in records]
            with patch.object(training,'_examples',return_value=iter(examples)),patch('arcade_ai_pet.channel_frontend.features',return_value=np.ones(260,dtype=np.float32)):
                arrays=training.make_arrays(root,'hp80',frontend='subtract')
            self.assertTrue((arrays['train']['x']==1).all())

    def test_training_resource_bounds_are_checked_before_loading_waveforms(self):
        with tempfile.TemporaryDirectory() as tmp:
            for values in ({'hidden':91},{'epochs':1001},{'hidden':True},{'epochs':0}):
                with self.subTest(values=values),self.assertRaisesRegex(ValueError,'training resource bounds'):
                    training.train_channel_candidate(Path('absent'),Path(tmp)/'out','hp80',**values)

    def test_checkpoint_rank_prefers_worst_channel_over_majority_average(self):
        from arcade_ai_pet import training as fit_module
        y=np.tile(np.arange(7),11);groups=np.array([0]*70+[1]*7)
        a=y.copy();a[70:73]=(a[70:73]+1)%7
        b=y.copy();b[:14]=(b[:14]+1)%7;b[70]=(b[70]+1)%7
        self.assertGreater(fit_module.metrics(y,a)['macro_f1'],fit_module.metrics(y,b)['macro_f1'])
        self.assertGreater(fit_module.validation_rank(y,b,groups),fit_module.validation_rank(y,a,groups))
        for invalid in (groups[:1],np.full(77,8),np.full(77,True)):
            with self.assertRaises(ValueError):fit_module.validation_rank(y,a,invalid)

    def test_noise_diversity_augments_only_train_and_leaves_validation_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            records=[dict(id=('a' if i==0 else 'b')*64,split=s,sha256=str(i)*64) for i,s in enumerate(('train','validation'))]
            (root/'manifest.json').write_text(json.dumps(dict(variants=1,records=records)))
            pcm=np.zeros(32000,dtype=np.int16);pcm[4000:12000]=(1000*np.sin(np.arange(8000)*.1)).astype(np.int16)
            examples=[(pcm,i,np.zeros(260,dtype=np.float32)) for i in range(2)]
            def arrays(replicas,mode='fresh'):
                with patch.object(training,'_examples',return_value=iter(examples)):
                    return training.make_arrays(root,'hp160',frontend='adaptive',noise_replicas=replicas,noise_mode=mode)
            base=arrays(1);fresh=arrays(3);repeat=arrays(3,'repeat')
            self.assertEqual(len(fresh['train']['y']),10);self.assertEqual(len(fresh['validation']['y']),8)
            np.testing.assert_array_equal(base['validation']['x'],fresh['validation']['x'])
            np.testing.assert_array_equal(base['validation']['x'],repeat['validation']['x'])
            indexes=[i for i,p in enumerate(fresh['train']['profiles']) if p=='noise']
            self.assertEqual(len(indexes),3)
            self.assertFalse(np.array_equal(fresh['train']['x'][indexes[0]],fresh['train']['x'][indexes[1]]))
            np.testing.assert_array_equal(repeat['train']['x'][indexes[0]],repeat['train']['x'][indexes[1]])
            again=arrays(3);np.testing.assert_array_equal(again['train']['x'],fresh['train']['x'])
        for kwargs in ({'noise_replicas':0},{'noise_replicas':9},{'noise_replicas':True},{'noise_mode':'device'}):
            with self.assertRaises(ValueError):training.make_arrays(Path('absent'),'hp160',**kwargs)

    def test_threshold_search_checks_integer_decision_boundaries_not_only_coarse_steps(self):
        from arcade_ai_pet.channel_robustness import PROFILES
        labels=np.tile(np.arange(7),len(PROFILES));profiles=[p for p in PROFILES for _ in range(7)]
        logits=np.zeros((len(labels),7),dtype=np.int64)
        logits[np.arange(len(labels)),labels]=5000
        # Reject the unknown margin 1279, retain the command margin 1400.
        logits[5]=0;logits[5,0]=1279
        logits[0]=0;logits[0,0]=1400
        threshold,rows,accepted=training.select_channel_threshold(labels,logits,profiles,'hp160')
        self.assertEqual(threshold,1280);self.assertTrue(accepted)
        self.assertTrue(all(r['macro_f1']==1 and r['command_recall']==1 and r['false_activation_rate']==0 for r in rows))
        with self.assertRaises(ValueError):training.select_channel_threshold(labels,logits[:1],profiles,'hp160')
        with self.assertRaises(ValueError):training.select_channel_threshold(labels,logits,['unknown']*len(labels),'hp160')

    def test_reselection_preserves_original_model_and_rejects_changed_weights_before_waveforms(self):
        import shutil
        from test_frozen import checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model';checkpoint(model)
            (root/'manifest.json').write_text('{}')
            digest=hashlib.sha256((model/'weights.npz').read_bytes()).hexdigest()
            metadata=dict(candidate='hp160',frontend_version='adaptive',noise_replicas=1,noise_mode='fresh',
                          weights_sha256=digest,dataset_manifest_sha256=hashlib.sha256(b'{}').hexdigest(),threshold_q10=0)
            (model/'validation.json').write_text(json.dumps(metadata));original=(model/'validation.json').read_bytes()
            y=np.tile(np.arange(7),8);profiles=[p for p in PROFILES for _ in range(7)]
            logits=np.zeros((56,7),dtype=np.int64);logits[np.arange(56),y]=5000
            logits[5]=0;logits[5,0]=1279;logits[0]=0;logits[0,0]=1400
            arrays={'validation':dict(x=np.zeros((56,260),dtype=np.float32),y=y,profiles=profiles)}
            with patch.object(training,'make_arrays',return_value=arrays),patch('arcade_ai_pet.training.scores',return_value=logits):
                result=training.reselect_threshold(model,root,root/'selected')
            self.assertTrue(result['accepted_validation']);self.assertEqual(result['threshold_q10'],1280)
            self.assertEqual(hashlib.sha256((root/'selected/weights.npz').read_bytes()).hexdigest(),digest)
            self.assertEqual((model/'validation.json').read_bytes(),original)
            with (model/'weights.npz').open('ab') as stream:stream.write(b'changed')
            with patch.object(training,'make_arrays',side_effect=AssertionError('must reject before waveforms')):
                with self.assertRaises(ValueError):training.reselect_threshold(model,root,root/'refused')
            self.assertFalse((root/'refused').exists())
