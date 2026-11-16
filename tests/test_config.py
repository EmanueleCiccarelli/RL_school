import unittest

from play2learn_lunarlander.config import (
    ENV_ID,
    MISSIONS,
    DEFAULT_PPO_PARAMS,
    VECNORM_PARAMS,
    DEFAULT_N_ENVS,
    DEFAULT_TIMESTEPS,
    HELIPAD_X_MIN,
    HELIPAD_X_MAX,
)


class TestConfig(unittest.TestCase):
    def test_env_id(self):
        self.assertEqual(ENV_ID, "LunarLander-v3")

    def test_missions_structure(self):
        self.assertIn("luna_standard", MISSIONS)
        self.assertIn("eco_lander", MISSIONS)
        self.assertIn("speedrun", MISSIONS)
        self.assertIn("tempesta_solare", MISSIONS)
        self.assertIn("pianeta_pesante", MISSIONS)
        self.assertIn("marte_leggero", MISSIONS)

        for key, m in MISSIONS.items():
            self.assertIn("title", m)
            self.assertIn("description", m)
            self.assertIn("goal", m)
            self.assertIn("env_kwargs", m)
            self.assertIn("gravity", m["env_kwargs"])

    def test_ppo_hyperparameters(self):
        self.assertEqual(DEFAULT_PPO_PARAMS["batch_size"], 128)
        self.assertEqual(DEFAULT_PPO_PARAMS["n_epochs"], 4)
        self.assertEqual(DEFAULT_PPO_PARAMS["gamma"], 0.995)
        self.assertEqual(DEFAULT_PPO_PARAMS["gae_lambda"], 0.98)
        self.assertEqual(DEFAULT_PPO_PARAMS["target_kl"], 0.03)
        self.assertGreater(DEFAULT_PPO_PARAMS["learning_rate"], 0)

    def test_vecnorm_params(self):
        self.assertFalse(VECNORM_PARAMS["norm_obs"])
        self.assertTrue(VECNORM_PARAMS["norm_reward"])
        self.assertEqual(VECNORM_PARAMS["clip_reward"], 10.0)

    def test_defaults(self):
        self.assertEqual(DEFAULT_N_ENVS, 8)
        self.assertEqual(DEFAULT_TIMESTEPS, 300_000)
        self.assertEqual(HELIPAD_X_MIN, -0.20)
        self.assertEqual(HELIPAD_X_MAX, 0.20)


if __name__ == "__main__":
    unittest.main()
