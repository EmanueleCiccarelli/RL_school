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

    def test_features_extractor_translation_invariance(self):
        box_space = gym.spaces.Box(low=-1000.0, high=1000.0, shape=(8,), dtype=np.float32)
        extractor = HillClimbFeaturesExtractor(box_space)

        # Due stati identici eccetto per le coordinate assolute x e y
        t1 = torch.tensor([[100.0, 50.0, 10.0, -2.0, 0.3, 0.5, 1.0, 1.0]], dtype=torch.float32)
        t2 = torch.tensor([[950.0, 350.0, 10.0, -2.0, 0.3, 0.5, 1.0, 1.0]], dtype=torch.float32)

        feat1 = extractor(t1)
        feat2 = extractor(t2)

        self.assertEqual(feat1.shape, (1, 7))
        self.assertEqual(feat2.shape, (1, 7))
        self.assertTrue(torch.allclose(feat1, feat2, atol=1e-5))

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
        # Verifica popolamento telemetria a fine episodio (allineato a Lunar Lander)
        wrapped.unwrapped.agent.dead = True
        obs, rew, term, trunc, info = wrapped.step(0)
        self.assertTrue(term or trunc)
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

    def test_classify_outcome_edge_cases(self):
        # 1. Distanza >= 800m su timeout (max_steps raggiunto senza morte) -> SUCCESSO
        obs_goal_timeout = np.array([850.0, 10.0, 2.0, 0.0, 0.1, 0.0, 1.0, 1.0], dtype=np.float32)
        succ, crash, tout, _ = classify_hill_climb_outcome(
            obs=obs_goal_timeout,
            original_reward=0.0,
            terminated=False,
            truncated=True,
        )
        self.assertTrue(succ, "Un agente che supera 800m allo scadere del tempo deve essere classificato come successo!")
        self.assertFalse(crash)
        self.assertFalse(tout)

        # 2. Blocco in salita / fosso (step_stuck_counter -> terminated=True, is_dead=False) -> TIMEOUT, NON CRASH
        obs_stuck = np.array([250.0, 10.0, 0.0, 0.0, 0.1, 0.0, 1.0, 1.0], dtype=np.float32)
        succ_s, crash_s, tout_s, _ = classify_hill_climb_outcome(
            obs=obs_stuck,
            original_reward=-100.0,
            terminated=True,
            truncated=False,
        )
        self.assertFalse(succ_s)
        self.assertFalse(crash_s, "Un blocco in salita senza morte non deve essere classificato come schianto!")
        self.assertTrue(tout_s)

        # 3. Angle wrapping su giro completo a 360 gradi
        full_circle_angle = float(2 * math.pi + 0.1)
        wrapped_ang = _wrap_angle(full_circle_angle)
        self.assertAlmostEqual(wrapped_ang, 0.1, places=5)

    def test_plot_hill_telemetry_zero_division_guard(self):
        with tempfile.TemporaryDirectory() as td:
            out_img = Path(td) / "telemetria_zero.png"
            # Simulazione di run dove tutti i conteggi esiti sono 0
            history_zeros = [
                {"distance": 100.0, "max_speed": 4.0, "airtime_pct": 5.0, "is_success": False, "is_crashed": False, "is_timeout": False}
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

        # 1. Unpacking a 2 tuple (compatibile con Lunar Lander)
        m, meta = fav
        self.assertEqual(m, fake_model)
        self.assertEqual(meta["mission"], "Test")

        # 2. Accesso ad attributi
        self.assertEqual(fav.model, fake_model)
        self.assertEqual(fav.metadata, fake_meta)
        self.assertEqual(fav.dir_path, fake_dir)

        # 3. Accesso a dizionario
        self.assertEqual(fav["metadata"], fake_meta)
        self.assertEqual(fav.get("mission"), "Test")


if __name__ == "__main__":
    unittest.main()
