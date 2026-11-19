"""Pinned ARM64 SDK must run on a stable standard ARM64 GitHub runner."""
from pathlib import Path
import unittest,re
ROOT=Path(__file__).resolve().parents[1]
class WasmCIContractTests(unittest.TestCase):
 def test_wasm_keeps_sdk_two_vm_checks_and_uses_stable_arm64_runner(self):
  text=(ROOT/'.github/workflows/tests.yml').read_text();job=text.split('\n  wasm:\n',1)[1]
  self.assertRegex(job,r'(?m)^    runs-on: macos-26$')
  self.assertIn('timeout-minutes: 15',job);self.assertIn('ref: 3bb9548846505eac303927b9064c6a8d0eb44f79',job)
  self.assertIn('python .cache/sdk/tools/bootstrap.py',job);self.assertIn('python -m unittest discover -s tests/integration -v',job)
  self.assertNotIn('continue-on-error',job);self.assertNotIn('xlarge',job)
