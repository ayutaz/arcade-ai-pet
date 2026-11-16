import importlib
import os
from pathlib import Path
import tempfile
import unittest
from arcade_ai_pet.build import build_pet
ROOT=Path(__file__).resolve().parents[2]
SDK=Path(os.environ.get('ARCADE_OS_ROOT',str(ROOT.parent/'scratch-arcade-os')))

class PetVmTests(unittest.TestCase):
    def test_real_package_starts_with_existing_fuel_and_two_vms_match_button_play(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=build_pet(SDK,Path(tmp).resolve()/'pet')
            decode=importlib.import_module('labos.package').decode
            Session=importlib.import_module('labos.sdk').Session
            native=importlib.import_module('labos.native')
            manifest,blob=decode(Path(result['package']).read_bytes())
            self.assertLessEqual(result['package_bytes'],16384);self.assertEqual(manifest['fuel'],50000)
            inputs=[(0,0),(32,0),(0,200)]+[(0,200)]*120+[(64,16)]
            actual=native.run_probe(native.build_probe(),[blob],inputs,capabilities=7)
            session=Session(blob,manifest);now=0;last_score=0
            try:
                self.assertEqual(len(actual),len(inputs))
                for row,(held,dt) in zip(actual,inputs,strict=True):
                    now+=dt;exited=session.frame(held,now)
                    self.assertEqual(row['status'],'exit' if exited else 'running')
                    self.assertEqual(row['rects'],[list(command) for command in session.commands])
                    if not exited:
                        for short in ('value','x','y','elapsed'):self.assertEqual(row[short],session.debug('debug_'+short))
                        self.assertLessEqual(len(session.commands),128)
                        last_score=session.debug('debug_value')&4095
                self.assertGreater(last_score,0)
            finally:session.close()
            self.assertTrue(session.closed)

    def test_simulated_voice_uses_real_imports_and_releases_a_160k_native_vm(self):
        import ctypes as C
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp).resolve();result=build_pet(SDK,base/'pet')
            decode=importlib.import_module('labos.package').decode;Session=importlib.import_module('labos.sdk').Session
            manifest,blob=decode(Path(result['package']).read_bytes())
            native=importlib.import_module('labos.native');native.build_probe()
            output=base/'vm.so'
            command=['clang','-O2','-shared','-fPIC','-std=c11','-I'+str(SDK/'prototypes/lab-os/native'),
                     '-I'+str(SDK/'downloads/os-research/wamr-upstream/core/iwasm/include'),
                     str(SDK/'prototypes/lab-os/native/guest_vm.c'),str(ROOT/'tests/fixtures/pet_vm_bridge.c'),
                     str(SDK/'build/lab-os-native/wamr/libiwasm.a'),'-lpthread','-lm','-o',str(output)]
            subprocess.run(command,check=True,capture_output=True)
            lib=C.CDLL(str(output));lib.pet_vm_open.argtypes=[C.c_void_p,C.c_uint];lib.pet_vm_open.restype=C.c_void_p
            lib.pet_vm_free.argtypes=[C.c_void_p];lib.pet_vm_frame.argtypes=[C.c_void_p,C.c_uint,C.c_uint]
            lib.pet_vm_stat.argtypes=[C.c_void_p,C.c_uint];lib.pet_vm_voice.argtypes=[C.c_void_p,C.c_int]
            self.assertTrue(lib.pet_runtime_start())
            class Audio:
                def __init__(self):self.state=0;self.position=0;self.label=6;self.records=self.replies=self.stops=0
                def call(self,op,a,b):
                    if op==4:return self.state
                    if op==7:return self.position
                    if op==2:self.state=2;self.position=0;self.records+=1;return 1
                    if op==9:self.position=0;return self.label
                    if op==10:self.state=1;self.replies+=1;return 1
                    if op==6:self.state=0;self.stops+=1;return 1
                    return -1
            try:
                for label in range(7):
                    buffer=C.create_string_buffer(blob);rig=lib.pet_vm_open(buffer,len(blob));self.assertTrue(rig)
                    backend=Audio();session=Session(blob,manifest,audio=backend)
                    try:
                        for held,now in ((0,0),(16,16),(0,32)):
                            self.assertFalse(session.frame(held,now));self.assertEqual(lib.pet_vm_frame(rig,held,0 if now==0 else 16),0)
                        backend.state=0;backend.position=32000;backend.label=label;lib.pet_vm_voice(rig,label)
                        session.frame(0,48);self.assertEqual(lib.pet_vm_frame(rig,0,16),0)
                        expected=(label+1)<<24 if label<5 else 0
                        self.assertEqual(session.debug('debug_value'),expected);self.assertEqual(lib.pet_vm_stat(rig,0),expected)
                        self.assertEqual(backend.records,1);self.assertEqual(lib.pet_vm_stat(rig,1),1)
                        self.assertEqual(backend.replies,1 if label<5 else 0);self.assertEqual(lib.pet_vm_stat(rig,2),backend.replies)
                        self.assertTrue(session.frame(64,64));self.assertEqual(lib.pet_vm_frame(rig,64,16),1)
                        self.assertEqual(backend.stops,1);self.assertEqual(lib.pet_vm_stat(rig,3),1)
                    finally:session.close();lib.pet_vm_free(rig)
            finally:lib.pet_runtime_stop()
