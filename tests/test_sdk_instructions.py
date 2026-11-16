from pathlib import Path
import re
import unittest
from arcade_ai_pet.build import SDK_COMMIT

ROOT=Path(__file__).resolve().parents[1]

class SdkInstructionsTests(unittest.TestCase):
    def test_the_user_build_guide_and_ci_use_the_same_sdk_as_the_builder(self):
        guide=(ROOT/'docs/pet-app.md').read_text()
        self.assertIn('`'+SDK_COMMIT+'`',guide)
        workflow=(ROOT/'.github/workflows/tests.yml').read_text()
        refs=re.findall(r'repository: ayutaz/scratch-arcade-os\s+ref: ([0-9a-f]{40})',workflow)
        self.assertEqual(refs,[SDK_COMMIT])
