import unittest
import numpy as np
import gymnasium as gym

from play2learn_lunarlander.environment import (
    _build_reward_caller,
    classify_lunar_lander_outcome,
    CustomRewardWrapper,
)


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

    def test_classify_lunar_lander_outcome(self):
        # Stato di successo perfetto:
        # x=0.0, y=0.05, vx=0.02, vy=-0.05, angle=0.01, w=0.0, leg_l=1.0, leg_r=1.0
        obs_success = np.array([0.0, 0.05, 0.02, -0.05, 0.01, 0.0, 1.0, 1.0], dtype=np.float32)
        success, crashed, timeout, off = classify_lunar_lander_outcome(
            obs=obs_success,
            original_reward=100.0,
            terminated=True,
            truncated=False,
        )
        self.assertTrue(success)
        self.assertFalse(crashed)
        self.assertFalse(timeout)
        self.assertFalse(off)

        # Un solo piede tocca terra (in bilico sul bordo): non deve essere successo
        obs_single_leg = np.array([0.15, 0.05, 0.02, -0.05, 0.01, 0.0, 1.0, 0.0], dtype=np.float32)
        success_sl, crashed_sl, _, _ = classify_lunar_lander_outcome(
            obs=obs_single_leg,
            original_reward=0.0,
            terminated=True,
            truncated=False,
        )
        self.assertFalse(success_sl)

        # Schianto (Gym penalty -100)
        obs_crash = np.array([0.0, 0.0, 0.5, -2.0, 1.5, 0.0, 0.0, 0.0], dtype=np.float32)
        success_cr, crashed_cr, _, _ = classify_lunar_lander_outcome(
            obs=obs_crash,
            original_reward=-100.0,
            terminated=True,
            truncated=False,
        )
        self.assertFalse(success_cr)
        self.assertTrue(crashed_cr)

    def test_custom_reward_wrapper_nan_raises_error(self):
        def bad_reward(obs, action):
            return float("nan")

        env = gym.make("LunarLander-v3")
        wrapped = CustomRewardWrapper(env, reward_fn=bad_reward)
        wrapped.reset(seed=42)
        with self.assertRaises(RuntimeError):
            wrapped.step(0)
        wrapped.close()


if __name__ == "__main__":
    unittest.main()

