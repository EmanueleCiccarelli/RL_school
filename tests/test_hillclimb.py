import unittest
import numpy as np
import gymnasium as gym

import play2learn_hillclimb
from play2learn_hillclimb.config import ENV_ID, MISSIONS
from play2learn_hillclimb.environment import (
    HillClimbRewardWrapper,
    classify_hill_climb_outcome,
)
from play2learn_hillclimb.interactive import gioca_single_player, gioca_vs_agente


class TestHillClimb(unittest.TestCase):
    def test_config_and_missions(self):
        self.assertEqual(ENV_ID, "HillClimbRacing-v0")
        self.assertIn("campagna_standard", MISSIONS)
        self.assertIn("deserto_dune", MISSIONS)
        self.assertIn("montagna_estrema", MISSIONS)
        self.assertIn("luna_bassa_gravita", MISSIONS)
        self.assertIn("autostrada_veloce", MISSIONS)

    def test_environment_gym_make_and_step(self):
        env = gym.make(ENV_ID, render_mode="rgb_array")
        obs, info = env.reset(seed=42)
        self.assertEqual(len(obs), 8)
        self.assertIn("score", info)

        obs, reward, term, trunc, info = env.step(1)
        self.assertEqual(len(obs), 8)
        self.assertIsInstance(reward, float)

        frame = env.render()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape[2], 3)
        env.close()

    def test_reward_wrapper_custom_reward(self):
        def my_reward(obs, action, is_success, is_crashed, is_timeout):
            dist = obs[0]
            vx = obs[2]
            return float(dist * 0.1 + vx)

        env = gym.make(ENV_ID, render_mode="rgb_array")
        wrapped = HillClimbRewardWrapper(env, reward_fn=my_reward)
        obs, _ = wrapped.reset(seed=42)
        obs, rew, term, trunc, info = wrapped.step(1)
        self.assertIsInstance(rew, float)
        self.assertIn("telemetry", info)
        self.assertIn("max_distance", info["telemetry"])
        wrapped.close()

    def test_reward_wrapper_nan_raises_error(self):
        def bad_reward(obs, action):
            return float("nan")

        env = gym.make(ENV_ID, render_mode="rgb_array")
        wrapped = HillClimbRewardWrapper(env, reward_fn=bad_reward)
        wrapped.reset(seed=42)
        with self.assertRaises(RuntimeError):
            wrapped.step(0)
        wrapped.close()

    def test_classify_outcome(self):
        # 1. Successo: distanza >= 800m
        obs_succ = np.array([850.0, 10.0, 5.0, 0.0, 0.1, 0.0, 1.0, 1.0], dtype=np.float32)
        succ, crash, tout, _ = classify_hill_climb_outcome(
            obs=obs_succ,
            original_reward=100.0,
            terminated=True,
            truncated=False,
        )
        self.assertTrue(succ)
        self.assertFalse(crash)

        # 2. Timeout: bloccato nel fosso
        obs_tout = np.array([120.0, 10.0, 0.0, 0.0, 0.1, 0.0, 1.0, 1.0], dtype=np.float32)
        succ_t, crash_t, tout_t, _ = classify_hill_climb_outcome(
            obs=obs_tout,
            original_reward=-100.0,
            terminated=False,
            truncated=True,
        )
        self.assertFalse(succ_t)
        self.assertTrue(tout_t)

    def test_interactive_smoke(self):
        ui = gioca_single_player(seed=42)
        self.assertIsNotNone(ui)
        html_str = str(getattr(ui, "data", ui))
        self.assertIn("Simulatore Manuale", html_str)

        ui_vs = gioca_vs_agente(model_or_result=None, seed=42)
        self.assertIsNotNone(ui_vs)
        html_vs = str(getattr(ui_vs, "data", ui_vs))
        self.assertIn("Sfida Hill Climb", html_vs)


if __name__ == "__main__":
    unittest.main()
