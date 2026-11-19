"""Host synthesis/ASR adapter wiring. No external model is loaded."""
from pathlib import Path
from types import SimpleNamespace
import tempfile,unittest,wave
from unittest.mock import Mock
import numpy as np
from arcade_ai_pet.host_teacher import source_callbacks

class Tensor:
    def __init__(self,value):self.value=value
    def detach(self):return self
    def float(self):return self
    def cpu(self):return self
    def numpy(self):return self.value

class HostTeacherTests(unittest.TestCase):
    def fixtures(self):
        runtime=Mock();runtime.synthesize.return_value=SimpleNamespace(audio=Tensor(np.ones((1,48000),dtype=np.float32)*.1),sample_rate=48000)
        factory=Mock(return_value='request');asr=Mock();asr.transcribe.return_value=(iter([SimpleNamespace(text='待って。')]),object())
        resample=Mock(side_effect=lambda values,rate:np.ascontiguousarray(values[::3],dtype=np.float32))
        return runtime,factory,asr,resample
    def test_teacher_uses_only_caption_no_reference_and_fixed_bounds(self):
        rt,factory,asr,resample=self.fixtures();synthesize,_=source_callbacks(rt,asr,request_factory=factory,resample=resample)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp).resolve()/'sample.wav';synthesize('まって','落ち着いた声',42,path)
            factory.assert_called_once_with(text='まって',caption='落ち着いた声',seed=42,no_ref=True,seconds=2.,num_steps=40)
            rt.synthesize.assert_called_once_with('request')
            with wave.open(str(path),'rb') as stream:self.assertEqual((stream.getnchannels(),stream.getsampwidth(),stream.getnframes()),(1,2,48000))
        asr.transcribe.assert_not_called()
    def test_asr_is_blind_and_receives_resampled_unit_float_audio(self):
        rt,factory,asr,resample=self.fixtures();_,recognize=source_callbacks(rt,asr,request_factory=factory,resample=resample)
        pcm=np.full(48000,16384,dtype=np.int16);self.assertEqual(recognize(48000,pcm),'待って。')
        args,kwargs=asr.transcribe.call_args;self.assertEqual(args[0].shape,(16000,));self.assertEqual(args[0].dtype,np.float32)
        self.assertTrue(np.all(args[0]==.5));self.assertTrue(np.all(pcm==16384))
        self.assertEqual(kwargs,dict(language='ja',beam_size=5,condition_on_previous_text=False,vad_filter=False,temperature=0.,max_new_tokens=32))
        factory.assert_not_called()
    def test_invalid_synthesis_never_writes_a_source(self):
        for values in (np.full((1,48000),np.nan),np.zeros((2,48000)),np.zeros((1,100000))):
            with self.subTest(shape=values.shape),tempfile.TemporaryDirectory() as tmp:
                rt,factory,asr,resample=self.fixtures();rt.synthesize.return_value.audio=Tensor(values)
                synthesize,_=source_callbacks(rt,asr,request_factory=factory,resample=resample);path=Path(tmp).resolve()/'bad.wav'
                with self.assertRaises(ValueError):synthesize('まって','声',42,path)
                self.assertFalse(path.exists())
    def test_existing_output_cannot_be_overwritten_or_trigger_teacher(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt,factory,asr,resample=self.fixtures();synthesize,_=source_callbacks(rt,asr,request_factory=factory,resample=resample)
            path=Path(tmp).resolve()/'existing.wav';path.write_bytes(b'existing')
            with self.assertRaises(ValueError):synthesize('まって','声',42,path)
            rt.synthesize.assert_not_called();self.assertEqual(path.read_bytes(),b'existing')

if __name__=='__main__':unittest.main()
