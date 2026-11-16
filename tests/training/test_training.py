import unittest
import numpy as np
from arcade_ai_pet.training import fit, scores, metrics, select_threshold


class TrainingTests(unittest.TestCase):
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
