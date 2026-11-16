from pathlib import Path
import tempfile
import unittest
import numpy as np
from arcade_ai_pet.replies import encode_reply,decode_reply,export_replies,REPLIES

class ReplyTests(unittest.TestCase):
    def test_resample_quantization_fade_and_decoded_ceiling(self):
        t=np.arange(24000)/16000
        pcm=np.rint(12000*np.sin(2*np.pi*440*t)+1000).astype(np.int16)
        encoded=encode_reply(pcm)
        self.assertEqual(encoded.dtype,np.int8);self.assertEqual(len(encoded),12000)
        decoded=decode_reply(encoded)
        self.assertEqual(len(decoded),24000);self.assertLessEqual(int(abs(decoded.astype(np.int32)).max()),3277)
        self.assertEqual(int(decoded[0]),0);self.assertEqual(int(decoded[-1]),0)
        self.assertGreater(float(np.corrcoef(pcm[100:-100],decoded[100:-100])[0,1]),.95)
        with self.assertRaises(ValueError):encode_reply(np.zeros(32001,dtype=np.int16))
        with self.assertRaises(ValueError):encode_reply(np.zeros(1000,dtype=np.int16))

    def test_five_texts_and_export_use_fresh_bounded_numeric_output(self):
        self.assertEqual(len(REPLIES),5)
        pcm=np.rint(np.sin(np.arange(8000)*.13)*10000).astype(np.int16)
        clips=[encode_reply(pcm) for _ in REPLIES]
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'assets';report=export_replies(clips,output)
            self.assertEqual(report['samples_hz'],8000)
            self.assertLessEqual(report['encoded_bytes'],80000)
            self.assertTrue((output/'replies.h').is_file())
            with self.assertRaises(ValueError):export_replies(clips,output)
            with self.assertRaises(ValueError):export_replies(clips[:4],Path(tmp)/'short')

    def test_synthesis_records_hashes_and_refuses_missing_engine_before_output(self):
        from unittest.mock import patch
        import wave
        from arcade_ai_pet.replies import synthesize
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=base/'engine';output=base/'assets'
            with self.assertRaises(ValueError):synthesize(root,output)
            self.assertFalse(output.exists())
            for name in ('bin/open_jtalk','dic/sys.dic','voice/m100/nitech_jp_atr503_m001.htsvoice'):
                file=root/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(b'fixture')
            def speak(argv,**kwargs):
                file=Path(argv[argv.index('-ow')+1]);pcm=np.rint(np.sin(np.arange(8000)*.1)*10000).astype('<i2')
                with wave.open(str(file),'wb') as stream:
                    stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(16000);stream.writeframes(pcm.tobytes())
            with patch('arcade_ai_pet.replies.subprocess.run',side_effect=speak) as call:
                report=synthesize(root,output)
            self.assertEqual(call.call_count,5);self.assertEqual(len(report['source_pcm_sha256']),5)
            self.assertEqual(len(report['voice_sha256']),64)
