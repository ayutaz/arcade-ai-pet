import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from arcade_ai_pet.build import build_pet

class BuildTests(unittest.TestCase):
    def test_bad_sdk_or_existing_output_never_builds(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp).resolve()/'new'
            with patch('arcade_ai_pet.build.load_sdk',side_effect=ValueError()):
                with self.assertRaises(ValueError):build_pet('wrong',output)
            self.assertFalse(output.exists())
    def test_build_stages_only_the_app_and_fixed_learned_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp).resolve()/'app';calls=[]
            def compile(folder):
                calls.append(folder)
                config=json.loads((folder/'app.json').read_text())
                self.assertEqual(config['id'],'ai-pet');self.assertEqual(config['capabilities'],['draw','speaker','microphone'])
                self.assertEqual({p.name for p in folder.iterdir()},{'app.c','app.json','behavior.c','behavior_model.h'})
                target=folder/'build';target.mkdir();(target/'app.labapp').write_bytes(b'fixture')
                return dict(package=str(target/'app.labapp'),package_bytes=7,device_tested=False)
            with patch('arcade_ai_pet.build.load_sdk',return_value=SimpleNamespace(build_project=compile)):
                result=build_pet('sdk',output)
                self.assertEqual(result['package_sha256'],hashlib.sha256(b'fixture').hexdigest())
                with self.assertRaises(ValueError):build_pet('sdk',output)
            self.assertEqual(len(calls),1)
