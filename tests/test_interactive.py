import unittest
import gymnasium as gym
from stable_baselines3 import PPO

from play2learn_lunarlander.interactive import (
    _extract_terrain_and_helipad,
    registra_traiettoria_agente,
    gioca_single_player,
    gioca_vs_agente,
)


class TestInteractive(unittest.TestCase):
    def test_terrain_extraction(self):
        env = gym.make("LunarLander-v3")
        env.reset(seed=42)
        pts, helipad = _extract_terrain_and_helipad(env)
        env.close()

        self.assertGreaterEqual(len(pts), 10)
        self.assertLess(helipad[0], helipad[1])
        self.assertGreater(helipad[2], 0)

    def test_agent_trajectory_recording(self):
        env = gym.make("LunarLander-v3")
        model = PPO("MlpPolicy", env, n_steps=64, batch_size=64)
        env.close()

        data = registra_traiettoria_agente(model=model, seed=42, max_steps=20)
        self.assertIn("terrain", data)
        self.assertIn("helipad", data)
        self.assertIn("steps", data)
        self.assertIn("stats", data)
        self.assertGreater(len(data["steps"]), 0)
        self.assertIn("x", data["steps"][0])
        self.assertIn("y", data["steps"][0])
        self.assertIn("action", data["steps"][0])

    def test_gioca_single_player_smoke(self):
        ui = gioca_single_player(seed=42)
        self.assertIsNotNone(ui)
        html_str = str(getattr(ui, "data", ui))
        self.assertIn("Simulatore Manuale", html_str)

    def test_gioca_vs_agente_signature_and_smoke(self):
        env = gym.make("LunarLander-v3")
        model = PPO("MlpPolicy", env, n_steps=64, batch_size=64)
        env.close()

        ui = gioca_vs_agente(
            model=model,
            seed=42,
            mission="luna_standard",
            env_kwargs={"gravity": -10.0},
        )
        self.assertIsNotNone(ui)
        html_str = str(getattr(ui, "data", ui))
        self.assertIn("Sfida Allunaggio", html_str)


if __name__ == "__main__":
    unittest.main()
