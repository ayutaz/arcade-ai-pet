import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import channel_training as training
from arcade_ai_pet.channel_robustness import PROFILES


class ChannelTrainingRecipeTests(unittest.TestCase):
    def test_training_recipe_records_per_profile_validation_without_evaluating_final_data(self):
        from arcade_ai_pet.training import fit
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'manifest.json').write_text('{}')
            x=np.zeros((56,260),dtype=np.float32);y=np.tile(np.arange(7),8)
            x[np.arange(56),y]=4
            # Profiles are interleaved by source in the real generator.
            profiles=[p for p in PROFILES for _ in range(7)]
            data={s:dict(x=x,y=y,profiles=profiles) for s in ('train','validation')}
            with patch.object(training,'make_arrays',return_value=data):
                result=training.train_channel_candidate(root,root/'out','hp80',epochs=1,frontend='subtract',hidden=88)
            self.assertEqual(result['frontend_version'],'subtract')
            self.assertEqual(result['weights_bytes'],23884)
            self.assertFalse(result['final_evaluated']);self.assertFalse(result['device_tested'])
            self.assertEqual(result['split_examples'],{'train':56,'validation':56})
            self.assertEqual(len(result['validation']),8)
            self.assertEqual({r['profile'] for r in result['validation']},set(PROFILES))
            self.assertTrue((root/'out/weights.npz').is_file())
            with self.assertRaises(ValueError):training.train_channel_candidate(root,root/'out','hp80',epochs=1)
