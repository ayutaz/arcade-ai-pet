"""Frozen numeric delivery must not regenerate platform-dependent math tables."""
from pathlib import Path
import hashlib,json,shutil,tempfile,unittest
from unittest.mock import patch
import numpy as np
from arcade_ai_pet import dscnn_assets as assets,logmel_native
from arcade_ai_pet.dscnn_freeze import verify,source_hashes
from arcade_ai_pet.dscnn_model import scores
from arcade_ai_pet.logmel_release import FrozenLogmelModel,verify_release
ROOT=Path(__file__).resolve().parents[1];MODEL=ROOT/'models/kws-logmel-v1'
TABLE_SHA='bd9ff5df6f6033179122c5f957bb6cbeaf1445d643dc1ef861153f6bfa0da110'

class FrozenReleaseLoaderTests(unittest.TestCase):
    def test_platform_table_regeneration_difference_does_not_change_frozen_runtime(self):
        # Reproduce the Linux failure on every host, without relaxing byte hashes.
        with patch.object(assets,'render_tables',return_value='different platform table'), \
             patch.object(logmel_native,'render_tables',return_value='not compilable generated table'):
            with self.assertRaisesRegex(ValueError,'frozen header changed'):verify(MODEL)
            meta,layers=verify_release(MODEL)
            with FrozenLogmelModel(MODEL) as engine:
                raw=np.zeros(32000,np.int16)
                self.assertEqual(engine.predict(engine.extract(raw))[0],6)
                self.assertEqual(engine.frontend.tables_sha256,TABLE_SHA)
                self.assertEqual(engine.metadata['weights_sha256'],meta['weights_sha256'])

    def test_corrupt_or_symlinked_frozen_inputs_are_refused_before_compilation(self):
        for name in ('weights.npz','logmel_tables.h','logmel_model.h','decision-frozen.json',
                     'decision-metadata.json','final-plan.json'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp)/'model';shutil.copytree(MODEL,folder)
                path=folder/name;path.write_bytes(path.read_bytes()+b'corrupt')
                with patch('subprocess.run') as compile,self.assertRaises(ValueError):FrozenLogmelModel(folder)
                compile.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'model';shutil.copytree(MODEL,folder);(folder/'logmel_tables.h').unlink()
            (folder/'logmel_tables.h').symlink_to(MODEL/'logmel_tables.h')
            with patch('subprocess.run') as compile,self.assertRaises(ValueError):FrozenLogmelModel(folder)
            compile.assert_not_called()

    def test_changed_frozen_source_is_not_accepted_as_a_portability_exception(self):
        with patch('arcade_ai_pet.dscnn_freeze.source_hashes',return_value={}):
            with self.assertRaisesRegex(ValueError,'source'):verify_release(MODEL)

    def test_frozen_c_frontend_and_integer_logits_match_in_both_layouts_without_mutating_inputs(self):
        before=source_hashes();meta,layers=verify_release(MODEL)
        with FrozenLogmelModel(MODEL) as engine:
            for hz in (0,220,970):
                raw=(np.sin(np.arange(32000)*2*np.pi*hz/16000)*2300).astype(np.int16);saved=raw.copy()
                values=engine.extract(raw);banked=engine.extract(raw,banked=True)
                np.testing.assert_array_equal(values,banked);np.testing.assert_array_equal(raw,saved)
                label,actual=engine.predict(values)
                np.testing.assert_array_equal(actual,scores(values[None,:],layers)[0]);self.assertIn(label,range(7))
            self.assertEqual(engine.scratch_bytes,26672)
        with self.assertRaises(ValueError):engine.extract(np.zeros(32000,np.int16))
        self.assertEqual(before,source_hashes());self.assertEqual(meta['source_sha256'],before)

    def test_delivered_headers_stay_byte_exact_even_if_local_formula_hash_differs(self):
        meta,layers=verify_release(MODEL);generated=assets.render(layers,meta['thresholds_q10'])
        self.assertEqual(hashlib.sha256(generated['logmel_model.h'].encode()).hexdigest(),meta['header_sha256']['logmel_model.h'])
        print(json.dumps(dict(scope='table_regeneration_diagnostic',frozen_tables_sha256=TABLE_SHA,
            local_generated_tables_sha256=hashlib.sha256(generated['logmel_tables.h'].encode()).hexdigest())))
