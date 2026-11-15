import unittest
import numpy as np

from play2learn_lunarlander.environment import _build_reward_caller


class TestEnvironment(unittest.TestCase):
    def test_reward_caller_5_params(self):
        def rew_fn(obs, action, is_success, is_crashed, is_timeout):
            return 10.0 if is_success else -1.0

        caller = _build_reward_caller(rew_fn)
        obs = np.zeros(8)
        self.assertEqual(caller(obs, 0, True, False, False), 10.0)
        self.assertEqual(caller(obs, 2, False, True, False), -1.0)

    def test_reward_caller_2_params(self):
        def rew_fn(obs, action):
            return float(action * 2)

        caller = _build_reward_caller(rew_fn)
        obs = np.zeros(8)
        self.assertEqual(caller(obs, 3, False, False, False), 6.0)

    def test_reward_caller_1_param(self):
        def rew_fn(obs):
            return float(obs[0])

        caller = _build_reward_caller(rew_fn)
        obs = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(caller(obs, 0, False, False, False), 0.5)

    def test_reward_caller_fallback(self):
        # Firma complessa con kwargs
        def rew_fn(obs, action, is_success, is_crashed, is_timeout, **extra):
            return 99.0

        caller = _build_reward_caller(rew_fn)
        obs = np.zeros(8)
        self.assertEqual(caller(obs, 2, False, False, False), 99.0)


if __name__ == "__main__":
    unittest.main()

