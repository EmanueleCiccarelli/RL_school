"""
Test Suite per Play2Learn Hill Climb Racing.
"""

import math
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import play2learn_hillclimb
from play2learn_hillclimb.agent import HillClimbFeaturesExtractor
from play2learn_hillclimb.config import ENV_ID, MISSIONS
from play2learn_hillclimb.environment import (
    HillClimbRewardWrapper,
    classify_hill_climb_outcome,
    _wrap_angle,
)
from play2learn_hillclimb.favorites import LoadedFavorite
from play2learn_hillclimb.visualizer import plot_hill_telemetry


class TestHillClimb(unittest.TestCase):
    def test_config_and_missions(self):
        self.assertEqual(ENV_ID, "hill_racing_env/HillRacing-v0")
        self.assertIn("campagna_standard", MISSIONS)
        self.assertIn("deserto_dune", MISSIONS)
        self.assertIn("montagna_estrema", MISSIONS)
        self.assertIn("colline_ripide", MISSIONS)
        self.assertIn("autostrada_veloce", MISSIONS)

    def test_environment_gym_make_and_step(self):
        env = gym.make(ENV_ID, render_mode="rgb_array")
        obs, info = env.reset(seed=42)
        self.assertIn("chassis_position", obs)

        obs, reward, term, trunc, info = env.step(1)
        self.assertIsInstance(reward, (int, float))

        frame = env.render()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape[2], 3)
        self.assertEqual(frame.shape[0], 720)
        self.assertEqual(frame.shape[1], 1280)
        env.close()

    def test_features_extractor_translation_invariance(self):
        box_space = gym.spaces.Box(low=-1000.0, high=1000.0, shape=(8,), dtype=np.float32)
        extractor = HillClimbFeaturesExtractor(box_space)

        # Due stati identici eccetto per le coordinate assolute x e y
        t1 = torch.tensor([[100.0, 50.0, 10.0, -2.0, 0.3, 0.5, 1.0, 1.0]], dtype=torch.float32)
        t2 = torch.tensor([[950.0, 350.0, 10.0, -2.0, 0.3, 0.5, 1.0, 1.0]], dtype=torch.float32)

        feat1 = extractor(t1)
        feat2 = extractor(t2)

        self.assertEqual(feat1.shape, (1, 64))
        self.assertEqual(feat2.shape, (1, 64))
        self.assertTrue(torch.allclose(feat1, feat2, atol=1e-5))

    def test_reward_wrapper_custom_reward(self):
        def my_reward(obs, action, is_success, is_crashed, is_timeout):
            dist = obs[0]
            vx = obs[2]
            return float(dist * 0.1 + vx)

        raw = gym.make(ENV_ID, render_mode="rgb_array")
        wrapped = HillClimbRewardWrapper(raw, reward_fn=my_reward)
        obs, _ = wrapped.reset(seed=42)
        self.assertEqual(len(obs), 8)

        obs, rew, term, trunc, info = wrapped.step(1)
        self.assertEqual(len(obs), 8)
        self.assertIsInstance(rew, float)
        wrapped.close()

    def test_reward_wrapper_nan_raises_error(self):
        def bad_reward(obs, action):
            return float("nan")

        raw = gym.make(ENV_ID, render_mode="rgb_array")
        wrapped = HillClimbRewardWrapper(raw, reward_fn=bad_reward)
        wrapped.reset(seed=42)
        with self.assertRaises((RuntimeError, ValueError)):
            wrapped.step(0)
        wrapped.close()

    def test_classify_outcome_edge_cases(self):
        # 1. Distanza >= 800m su timeout -> SUCCESSO
        obs_goal_timeout = np.array([850.0, 10.0, 2.0, 0.0, 0.1, 0.0, 1.0, 1.0], dtype=np.float32)
        succ, crash, tout, _ = classify_hill_climb_outcome(
            obs=obs_goal_timeout,
            original_reward=0.0,
            terminated=False,
            truncated=True,
        )
        self.assertTrue(succ)
        self.assertFalse(crash)
        self.assertFalse(tout)

        # 2. Blocco in salita senza morte (distanza < 800m) -> TIMEOUT
        obs_stuck = np.array([250.0, 10.0, 0.0, 0.0, 0.1, 0.0, 1.0, 1.0], dtype=np.float32)
        succ_s, crash_s, tout_s, _ = classify_hill_climb_outcome(
            obs=obs_stuck,
            original_reward=-100.0,
            terminated=True,
            truncated=False,
        )
        self.assertFalse(succ_s)
        self.assertTrue(crash_s or tout_s)

        # 3. Angle wrapping
        full_circle = float(2 * math.pi + 0.1)
        wrapped_ang = _wrap_angle(full_circle)
        self.assertAlmostEqual(wrapped_ang, 0.1, places=5)

    def test_plot_hill_telemetry_zero_division_guard(self):
        with tempfile.TemporaryDirectory() as td:
            out_img = Path(td) / "telemetria_zero.png"
            history_zeros = [
                {"distance": 100.0, "max_speed": 4.0, "airtime_pct": 5.0, "reward": 10.0}
                for _ in range(20)
            ]
            saved = plot_hill_telemetry(history_zeros, output_path=out_img)
            self.assertIsNotNone(saved)
            self.assertTrue(out_img.exists())

    def test_loaded_favorite_hybrid_access(self):
        fake_model = "fake_ppo_model"
        fake_meta = {"mission": "Test", "eval_metrics": {"eval_success_rate": 85.0}}
        fake_dir = Path("/fake/dir")
        fake_entry = {"id": 1, "nome": "Mio Test"}

        fav = LoadedFavorite(model=fake_model, metadata=fake_meta, dir_path=fake_dir, entry=fake_entry)

        # 1. Tupla unpack
        m, meta = fav
        self.assertEqual(m, fake_model)
        self.assertEqual(meta["mission"], "Test")

        # 2. Accesso attributi
        self.assertEqual(fav.model, fake_model)
        self.assertEqual(fav.metadata, fake_meta)

        # 3. Accesso dizionario
        self.assertEqual(fav["metadata"], fake_meta)
        self.assertEqual(fav.get("mission"), "Test")


if __name__ == "__main__":
    unittest.main()
