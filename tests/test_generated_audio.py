"""生成音源を独立ASRで点検する契約。偽音声を品質の実測とは扱わない。"""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
import wave
from unittest.mock import Mock
import numpy as np
from arcade_ai_pet.generated_audio import audit_generated_wave, normalize_reading


class GeneratedAudioAuditTests(unittest.TestCase):
    def source(self, root, *, rate=16000, seconds=.2, channels=1, silent=False):
        path=Path(root).resolve()/'source.wav'
        pcm=np.zeros(round(rate*seconds),dtype='<i2') if silent else np.rint(2000*np.sin(np.arange(round(rate*seconds))*.1)).astype('<i2')
        with wave.open(str(path),'wb') as stream:
            stream.setnchannels(channels);stream.setsampwidth(2);stream.setframerate(rate)
            stream.writeframes(np.repeat(pcm,channels).tobytes())
        return path

    def test_blind_asr_agreement_keeps_nominal_label_and_hash_without_raw_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.source(tmp);backend=Mock(return_value='待って。')
            result=audit_generated_wave(path,prompt='まって',recognize=backend)
            self.assertTrue(result['content_check_passed'])
            self.assertEqual(result['label'],2)
            self.assertEqual(result['wave_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            rate,pcm=backend.call_args.args
            self.assertEqual(rate,16000);self.assertEqual(pcm.shape,(3200,));self.assertEqual(pcm.dtype,np.int16)
            self.assertEqual(backend.call_args.kwargs,{})  # never pass expected text to the recognizer
            self.assertEqual(result['normalized_transcript'],'まって')
            self.assertFalse(result['human_transcript_verified'])
            self.assertNotIn(str(path),json.dumps(result));self.assertNotIn('samples',result)

    def test_different_word_negation_or_empty_recognition_never_relabels_a_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.source(tmp)
            for text in ('おいで','まってない','', 'あそぼおいで'):
                result=audit_generated_wave(path,prompt='あそぼ',recognize=lambda *args:text)
                self.assertFalse(result['content_check_passed']);self.assertEqual(result['label'],0)
                self.assertEqual(result['status'],'mismatch')

    def test_fixed_reading_variants_are_exact_and_do_not_use_substring_or_fuzzy_matching(self):
        self.assertEqual(normalize_reading(' 「遊ぼう！」 '),'あそぼ')
        self.assertEqual(normalize_reading('マッテ'),'まって')
        self.assertEqual(normalize_reading('お休み。'),'おやすみ')
        self.assertNotEqual(normalize_reading('あそぶ'),'あそぼ')
        self.assertNotEqual(normalize_reading('遊ぼうじゃない'),'あそぼ')

    def test_invalid_wave_prompt_and_silence_fail_before_asr(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend=Mock(return_value='あそぼ')
            for settings in ({'seconds':2.1},{'channels':2},{'silent':True}):
                path=self.source(tmp,**settings)
                with self.assertRaises(ValueError):audit_generated_wave(path,prompt='あそぼ',recognize=backend)
            path=self.source(tmp)
            with self.assertRaises(ValueError):audit_generated_wave(path,prompt='unknown private prompt',recognize=backend)
            alias=Path(tmp)/'alias.wav';alias.symlink_to(path)
            with self.assertRaises(ValueError):audit_generated_wave(alias,prompt='あそぼ',recognize=backend)
            backend.assert_not_called()

    def test_recognizer_failure_is_rejected_without_exception_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.source(tmp)
            for backend in (Mock(side_effect=RuntimeError('private exception text')), Mock(return_value={'secret':'private'})):
                result=audit_generated_wave(path,prompt='あそぼ',recognize=backend)
                self.assertFalse(result['content_check_passed'])
                self.assertEqual(result['status'],'recognizer_error')
                self.assertNotIn('private',json.dumps(result))
                self.assertIsNone(result['transcript'])

    def test_backend_mutation_does_not_change_audited_source_or_its_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.source(tmp);before=path.read_bytes()
            def backend(rate,pcm):pcm[:]=0;return 'こんにちは'
            result=audit_generated_wave(path,prompt='こんにちは',recognize=backend)
            self.assertTrue(result['content_check_passed']);self.assertEqual(result['label'],5)
            self.assertGreater(result['peak'],0);self.assertEqual(path.read_bytes(),before)

    def test_content_agreement_does_not_claim_license_or_dataset_training_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=audit_generated_wave(self.source(tmp),prompt='あそぼ',recognize=lambda *args:'遊ぼう')
            self.assertTrue(result['content_check_passed'])
            self.assertNotIn('accepted_for_training',result)


if __name__=='__main__':unittest.main()
