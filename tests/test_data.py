import unittest
import numpy as np
from arcade_ai_pet.data import augment, read_pcm, trim_silence, create, SOURCES


class DataTests(unittest.TestCase):
    def test_source_cache_is_hash_checked_and_output_is_never_overwritten(self):
        import json,tempfile,wave
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=base/'engine'
            for directory in ('bin','dic','voice/test'):(root/directory).mkdir(parents=True)
            (root/'bin/open_jtalk').write_bytes(b'engine fixture');(root/'dic/sys.dic').write_bytes(b'dictionary')
            (root/'voice/test/test.htsvoice').write_bytes(b'voice')
            def synth(argv,**kwargs):
                with wave.open(argv[argv.index('-ow')+1],'wb') as stream:
                    stream.setparams((1,2,16000,0,'NONE','not compressed'))
                    stream.writeframes((1000*np.sin(np.arange(8000)*.1)).astype('<i2').tobytes())
            with patch('arcade_ai_pet.data.subprocess.run',side_effect=synth) as command:
                first=create(base/'first',root,variants=1)
                count=command.call_count
                second=create(base/'second',root,variants=1,source_cache=base/'first')
                self.assertEqual(command.call_count,count);self.assertEqual(first,second)
                with self.assertRaises(ValueError):create(base/'second',root)
                record=json.loads((base/'first/manifest.json').read_text())['records'][0]
                (base/'first/sources'/(record['id']+'.wav')).write_bytes(b'changed')
                with self.assertRaises(ValueError):create(base/'third',root,source_cache=base/'first')

    def test_trim_keeps_speech_and_context_but_never_truncates_long_speech(self):
        x=np.zeros(48000,dtype=np.int16);x[16000:24000]=1000
        actual=trim_silence(x)
        self.assertEqual(np.count_nonzero(actual),8000);self.assertLess(len(actual),32000)
        with self.assertRaises(ValueError):trim_silence(np.full(40000,1000,dtype=np.int16))
        with self.assertRaises(ValueError):trim_silence(np.zeros(32000,dtype=np.int16))

    def test_augmentation_is_deterministic_fixed_size_and_bounded(self):
        x=(2000*np.sin(2*np.pi*400*np.arange(8000)/16000)).astype(np.int16)
        a=augment(x,123);b=augment(x,123)
        self.assertEqual(a.shape,(32000,));self.assertEqual(a.dtype,np.int16)
        np.testing.assert_array_equal(a,b)
        self.assertFalse(np.array_equal(a,augment(x,124)))
        self.assertLessEqual(abs(a.astype(np.int32)).max(),32767)
        with self.assertRaises(ValueError):augment(np.ones(40000,dtype=np.int16),1)

    def test_wav_requires_mono_pcm16_and_resamples_by_explicit_rate(self):
        import tempfile,wave
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'test.wav'
            with wave.open(str(file),'wb') as stream:
                stream.setparams((1,2,48000,0,'NONE','not compressed'))
                stream.writeframes(np.ones(4800,dtype='<i2').tobytes())
            self.assertEqual(read_pcm(file).shape,(1600,))
            with wave.open(str(file),'wb') as stream:
                stream.setparams((2,2,16000,0,'NONE','not compressed'));stream.writeframes(bytes(100))
            with self.assertRaises(ValueError):read_pcm(file)

    def test_labels_cover_five_commands_and_diverse_unknowns(self):
        self.assertEqual(set(SOURCES),set(range(6)))
        self.assertGreaterEqual(len(SOURCES[5]),15)
        self.assertEqual([SOURCES[i][0] for i in range(5)],['あそぼ','おいで','まって','おやすみ','ありがとう'])
