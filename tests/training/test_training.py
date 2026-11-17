import unittest
import numpy as np
from arcade_ai_pet.training import fit, scores, metrics, select_threshold
from arcade_ai_pet.training import run


class TrainingTests(unittest.TestCase):
    def test_validation_only_recipe_never_scores_test_and_preserves_requested_width(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from arcade_ai_pet import training
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);(base/'manifest.json').write_text('{}')
            x=np.zeros((7,260),dtype=np.float32);x[np.arange(7),np.arange(7)]=4;y=np.arange(7)
            test=x.copy();test[:,259]=15
            data=dict(train=dict(x=x,y=y),validation=dict(x=x,y=y),test=dict(x=test,y=y))
            original=training.scores
            def guarded(values,layers):
                if np.any(values[:,259]==15):raise AssertionError('held-out test was scored')
                return original(values,layers)
            with patch('arcade_ai_pet.training.load_dataset',return_value=data),patch('arcade_ai_pet.training.scores',side_effect=guarded):
                result=run(base,base/'output',epochs=2,hidden=16,validation_only=True)
            self.assertFalse(result['test_evaluated']);self.assertNotIn('test',result)
            self.assertTrue((base/'output/validation.json').exists())
            with np.load(base/'output/weights.npz',allow_pickle=False) as archive:self.assertEqual(archive['weight0'].shape,(16,260))
    def test_tiny_training_is_reproducible_and_integer_model_separates_inputs(self):
        x=np.zeros((40,260),dtype=np.float32);x[:20,0]=-2;x[20:,0]=2
        y=np.array([0]*20+[1]*20)
        a=fit(x,y,x,y,seed=42,epochs=60,hidden=16)
        b=fit(x,y,x,y,seed=42,epochs=60,hidden=16)
        self.assertEqual(a['validation_accuracy'],1)
        for left,right in zip(a['layers'],b['layers'],strict=True):
            np.testing.assert_array_equal(left['weight'],right['weight'])
            np.testing.assert_array_equal(left['bias'],right['bias'])
        self.assertEqual(scores(x,a['layers']).argmax(axis=1).tolist(),y.tolist())

    def test_confusion_metrics_and_validation_only_threshold(self):
        report=metrics(np.arange(7),np.arange(7))
        self.assertEqual(report['macro_f1'],1);self.assertEqual(report['false_activation_rate'],0)
        logits=np.zeros((7,7),dtype=np.int32);logits[np.arange(7),np.arange(7)]=4096
        threshold=select_threshold(np.arange(7),logits)
        self.assertGreaterEqual(threshold,0)
        self.assertEqual(metrics(np.array([5,6]),np.array([0,1]))['false_activation_rate'],1)

    def test_invalid_training_input_is_rejected(self):
        x=np.zeros((7,260),dtype=np.float32);y=np.arange(7)
        for invalid in (x[:,:2],np.full((7,260),np.nan)):
            with self.assertRaises(ValueError):fit(invalid,y,x,y,epochs=1)

    def test_88_hidden_units_fit_the_existing_24kib_budget_and_larger_models_are_refused(self):
        x=np.zeros((7,260),dtype=np.float32);x[np.arange(7),np.arange(7)]=4;y=np.arange(7)
        result=fit(x,y,x,y,seed=42,epochs=2,hidden=88)
        self.assertEqual(result['layers'][0]['weight'].shape,(88,260))
        size=sum(l['weight'].nbytes+l['bias'].nbytes+4 for l in result['layers'])
        self.assertEqual(size,23884);self.assertLessEqual(size,24576)
        with self.assertRaises(ValueError):fit(x,y,x,y,epochs=1,hidden=91)
