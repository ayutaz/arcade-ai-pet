"""モデル凍結後だけの最終条件生成。開発条件は再利用しない。"""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from arcade_ai_pet import final_corpus
from arcade_ai_pet import channel_freeze
import test_neural_corpus as source_fixture
import test_channel_freeze as freeze_fixture


class FinalCorpusTests(unittest.TestCase):
    def frozen(self,root):
        model,dev,arrays=freeze_fixture.ChannelFreezeTests().fixture(root)
        with patch.object(channel_freeze,'make_arrays',return_value=arrays):
            channel_freeze.freeze_channel(model,dev,root/'frozen')
        return root/'frozen'

    def test_only_final_conditions_generate_after_valid_model_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();frozen=self.frozen(root);output=root/'final'
            synthesize,recognize,calls=source_fixture.NeuralCorpusTests().provider()
            final_corpus.generate_final(source_fixture.plan(),output,frozen=frozen,synthesize=synthesize,recognize=recognize)
            self.assertEqual(len(calls),21);self.assertEqual({r[1] for r in calls},{'柔らかくゆっくりした声。'})
            self.assertFalse((output/'manifest.json').exists())
            qualified=json.loads((output/'qualified-sources.json').read_text())
            self.assertEqual(qualified['role'],'final');self.assertEqual(qualified['model_binding']['weights_sha256'],hashlib.sha256((frozen/'weights.npz').read_bytes()).hexdigest())
            final_corpus.finalize_final(output,frozen=frozen)
            manifest=json.loads((output/'manifest.json').read_text());self.assertEqual(manifest['role'],'final')
            self.assertEqual(manifest['model_binding'],qualified['model_binding'])
            self.assertEqual(len(calls),21)

    def test_changed_weights_refuse_before_synthesis_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();frozen=self.frozen(root);callback=Mock()
            with (frozen/'weights.npz').open('ab') as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):final_corpus.generate_final(source_fixture.plan(),root/'final',frozen=frozen,synthesize=callback,recognize=callback)
            callback.assert_not_called();self.assertFalse((root/'final').exists())

    def test_changed_frozen_metadata_refuses_final_features_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();frozen=self.frozen(root);output=root/'final'
            synthesize,recognize,calls=source_fixture.NeuralCorpusTests().provider()
            final_corpus.generate_final(source_fixture.plan(),output,frozen=frozen,synthesize=synthesize,recognize=recognize)
            path=frozen/'channel-frozen.json';metadata=json.loads(path.read_text());metadata['threshold_q10']=1;path.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):final_corpus.finalize_final(output,frozen=frozen)
            self.assertFalse((output/'manifest.json').exists());self.assertFalse((output/'train.npz').exists())
            self.assertEqual(len(calls),21)


if __name__=='__main__':unittest.main()
