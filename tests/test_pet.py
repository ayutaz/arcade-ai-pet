import unittest
from arcade_ai_pet.pet import Pet

class PetTests(unittest.TestCase):
    def test_five_voice_actions_button_fallback_and_unknown(self):
        pet=Pet(seed=17)
        self.assertEqual(pet.mode,'idle')
        for action,mode in ((0,'play'),(1,'come'),(2,'wait'),(3,'sleep'),(4,'happy')):
            before=pet.reply_serial;pet.command(action)
            self.assertEqual(pet.mode,mode);self.assertEqual(pet.reply,action)
            self.assertEqual(pet.reply_serial,before+1)
        snapshot=pet.snapshot()
        for ignored in (5,6,-1):pet.command(ignored);self.assertEqual(snapshot,pet.snapshot())
        pet.button(32);self.assertEqual(pet.mode,'play')
        for key in (1,2,4,8):pet.button(key)
        pet.button(64);self.assertTrue(pet.closed)
        snapshot=pet.snapshot();pet.command(0);pet.tick(100);self.assertEqual(pet.snapshot(),snapshot)

    def test_learned_companion_collects_and_pause_sleep_stop_time(self):
        pet=Pet(seed=17);pet.command(0)
        for _ in range(120):pet.tick(200)
        self.assertGreater(pet.world.score,0)
        self.assertEqual(pet.world.steps,120)
        pet.command(2);snapshot=pet.world.snapshot();pet.tick(1000);self.assertEqual(pet.world.snapshot(),snapshot)
        pet.command(3);pet.tick(1000);self.assertEqual(pet.world.snapshot(),snapshot)
        with self.assertRaises(ValueError):pet.tick(60001)
        with self.assertRaises(ValueError):pet.command(True)

    def test_player_has_separate_score_and_cannot_cross_walls(self):
        pet=Pet(seed=17);pet.command(0);pet.world.walls=set()
        pet.player=(6,5);pet.world.goal=(7,5)
        pet.button(1);self.assertEqual(pet.player,(7,5));self.assertEqual(pet.player_score,1)
        pet.button(1);self.assertEqual(pet.player,(7,5));self.assertEqual(pet.player_score,1)
        pet.world.walls={(6,5)};pet.button(2);self.assertEqual(pet.player,(7,5))
