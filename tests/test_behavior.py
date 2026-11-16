import unittest
import numpy as np
from arcade_ai_pet.behavior import World, teacher, observation, benchmark, random_action, greedy_action


class BehaviorTests(unittest.TestCase):
    def test_seed_replay_bounded_moves_and_collect_once(self):
        a=World(17);b=World(17)
        self.assertEqual(a.snapshot(),b.snapshot())
        for _ in range(120):
            action=teacher(a);self.assertIn(action,range(4))
            a.step(action);b.step(action)
            self.assertEqual(a.snapshot(),b.snapshot())
            self.assertNotIn((a.x,a.y),a.walls)
            self.assertEqual(observation(a).shape,(6,))
            self.assertLessEqual(float(abs(observation(a)).max()),1)
        self.assertGreater(a.score,10)
        with self.assertRaises(ValueError):a.step(4)
        with self.assertRaises(ValueError):World(True)

    def test_teacher_and_seeded_random_have_independent_fair_rollouts(self):
        learned=benchmark(teacher,seeds=range(20000,20008),steps=120)
        random=benchmark(random_action,seeds=range(20000,20008),steps=120)
        self.assertGreater(learned['mean_score'],random['mean_score'])
        self.assertEqual(random,benchmark(random_action,seeds=range(20000,20008),steps=120))
        self.assertEqual(learned['episodes'],8)
        with self.assertRaises(ValueError):benchmark(teacher,seeds=[1,1],steps=120)
        with self.assertRaises(ValueError):benchmark(greedy_action,seeds=[1],steps=0)
