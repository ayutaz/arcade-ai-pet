import ctypes as C
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from arcade_ai_pet.pet import Pet
ROOT=Path(__file__).resolve().parents[1]

class PetNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.temp.cleanup)
        base=Path(cls.temp.name)
        for file in ('app.c',):shutil.copyfile(ROOT/'app'/file,base/file)
        shutil.copyfile(ROOT/'native/behavior.c',base/'behavior.c')
        shutil.copyfile(ROOT/'models/behavior-v1/behavior_model.h',base/'behavior_model.h')
        output=base/'pet.so'
        run=subprocess.run(['clang','-O2','-shared','-fPIC','-std=c11','-Wall','-Wextra','-Werror',
                            '-I'+str(base),'-I'+str(ROOT/'tests/fixtures/pet_sdk'),str(ROOT/'tests/fixtures/pet_bridge.c'),'-o',str(output)],capture_output=True,text=True)
        if run.returncode:raise AssertionError(run.stderr)
        cls.lib=C.CDLL(str(output))
        cls.lib.test_tick.argtypes=[C.c_uint,C.c_uint]
    def setUp(self):self.lib.test_init()
    def test_audio_capture_recognizes_five_commands_and_never_acts_on_unknown(self):
        for command in range(7):
            self.lib.test_init();self.lib.test_tick(16,0);self.assertEqual(self.lib.test_stat(0),1)
            self.lib.test_tick(0,16);self.lib.test_voice(command);self.lib.test_tick(0,16)
            self.assertEqual(self.lib.debug_value()>>24,command+1 if command<5 else 0)
            self.assertEqual(self.lib.test_stat(1),0)
            self.lib.test_tick(0,16)
            self.assertEqual(self.lib.test_stat(1),1 if command<5 else 0)
            self.assertEqual(self.lib.test_tick(64,16),1)
            self.assertEqual(self.lib.test_stat(2),1);self.assertEqual(self.lib.test_stat(3),0)
    def test_c_world_and_learned_actions_match_python_on_unseen_boards(self):
        for seed in range(20000,20016):
            self.lib.test_init();self.lib.test_seed(seed);self.lib.test_action(0)
            pet=Pet(seed=seed);pet.command(0)
            for _ in range(120):
                pet.tick(200);self.lib.test_tick(0,200)
                self.assertEqual(self.lib.debug_value()&4095,pet.world.score)
                self.assertEqual(self.lib.debug_x()&255,pet.world.x)
                self.assertEqual(self.lib.debug_y()&255,pet.world.y)
                self.assertEqual(self.lib.debug_elapsed(),pet.world.steps)
            self.assertEqual(self.lib.test_stat(3),0)
    def test_button_fallback_pause_sleep_and_priority_exit_match_state_model(self):
        pet=Pet()
        for held in (32,0,1,0,8,0,2,0):
            pet.button(held);pet.tick(200);self.lib.test_tick(held,200)
        self.assertEqual(self.lib.debug_x()>>8,pet.player[0]);self.assertEqual(self.lib.debug_y()>>8,pet.player[1])
        self.assertEqual(self.lib.test_tick(64|16,16),1)
        self.assertEqual(self.lib.test_stat(0),0)

    def test_classification_defers_reply_until_fresh_tick_and_back_cancels_pending_reply(self):
        for command in range(5):
            self.lib.test_init();self.lib.test_tick(16,0);self.lib.test_tick(0,16)
            self.lib.test_voice(command);self.lib.test_tick(0,16)
            self.assertEqual(self.lib.debug_value()>>24,command+1)
            self.assertEqual(self.lib.test_stat(1),0,'classification must not start a reply with its cached timestamp')
            self.lib.test_tick(0,16);self.assertEqual(self.lib.test_stat(1),1)
            self.lib.test_tick(0,16);self.assertEqual(self.lib.test_stat(1),1)
        self.lib.test_init();self.lib.test_tick(16,0);self.lib.test_tick(0,16);self.lib.test_voice(0);self.lib.test_tick(0,16)
        self.assertEqual(self.lib.test_tick(64,16),1);self.assertEqual(self.lib.test_stat(1),0);self.assertEqual(self.lib.test_stat(2),1)
