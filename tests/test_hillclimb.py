"""
Test Suite e Regression Tests per Play2Learn Hill Climb Racing.
Copre:
  - Test originali di configurazione, ambiente, feature extractor, wrapper e preferiti
  - Regressioni fisica Box2D (masse, sospensioni Nyquist, rim_body, leak wallBody, anti-flickering)
  - Regressioni determinismo Perlin e integrità coordinate post-rendering
  - Regressioni API studente, budget temporale 800m, terminazione su successo, preferiti, ZIP e reportistica
"""

import inspect
import math
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

import play2learn_hillclimb
from hill_racing_env.envs import hill_racing
from hill_racing_env.envs.hill_racing import ContactListener
from play2learn_core.favorites import BaseFavoritesManager
from play2learn_hillclimb import favorites as hc_favorites
from play2learn_hillclimb.agent import HillClimbFeaturesExtractor
from play2learn_hillclimb.config import ENV_ID, MISSIONS, TARGET_DISTANCE_SUCCESS
from play2learn_hillclimb.environment import (
    HillClimbRewardWrapper,
    _wrap_angle,
    classify_hill_climb_outcome,
)
from play2learn_hillclimb.evaluator import evaluate_and_record
from play2learn_hillclimb.favorites import LoadedFavorite
from play2learn_hillclimb.reporting import genera_report_esperimento
from play2learn_hillclimb.storage import comprimi_archivio, save_run_metadata
from play2learn_hillclimb.visualizer import plot_hill_telemetry


class TestHillClimb(unittest.TestCase):
    # =========================================================================
    # TEST BASE ORIGINALI
    # =========================================================================

    def test_config_and_missions(self):
        self.assertEqual(ENV_ID, "hill_racing_env/HillRacing-v0")
        self.assertIn("campagna_standard", MISSIONS)
        self.assertIn("deserto_dune", MISSIONS)
        self.assertIn("montagna_estrema", MISSIONS)
        self.assertIn("colline_ripide", MISSIONS)
        self.assertIn("autostrada_veloce", MISSIONS)

    def test_environment_gym_make_and_step(self):
        env = gym.make(ENV_ID, render_mode="rgb_array")
        try:
            obs, info = env.reset(seed=42)
            self.assertIn("chassis_position", obs)

            obs, reward, term, trunc, info = env.step(1)
            self.assertIsInstance(reward, (int, float))

            frame = env.render()
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (720, 1280, 3))
            self.assertTrue(frame.flags["C_CONTIGUOUS"])
        finally:
            env.close()

    def test_features_extractor_translation_invariance(self):
        box_space = gym.spaces.Box(low=-1000.0, high=1000.0, shape=(8,), dtype=np.float32)
        extractor = HillClimbFeaturesExtractor(box_space)

        t1 = torch.tensor([[100.0, 50.0, 10.0, -2.0, 0.3, 0.5, 1.0, 1.0]], dtype=torch.float32)
        t2 = torch.tensor([[950.0, 350.0, 10.0, -2.0, 0.3, 0.5, 1.0, 1.0]], dtype=torch.float32)

        feat1 = extractor(t1)
        feat2 = extractor(t2)

        self.assertEqual(feat1.shape, feat2.shape)
        self.assertTrue(torch.allclose(feat1, feat2, atol=1e-5))

    def test_reward_wrapper_custom_reward(self):
        def my_reward(obs, action, is_success, is_crashed, is_timeout):
            return float(obs[0] * 0.1 + obs[2])

        raw = gym.make(ENV_ID, render_mode=None)
        wrapped = HillClimbRewardWrapper(raw, reward_fn=my_reward)
        try:
            obs, _ = wrapped.reset(seed=42)
            self.assertGreaterEqual(len(obs), 8)

            obs, rew, term, trunc, info = wrapped.step(1)
            self.assertGreaterEqual(len(obs), 8)
            self.assertIsInstance(rew, float)
        finally:
            wrapped.close()

    def test_reward_wrapper_nan_raises_error(self):
        def bad_reward(obs, action):
            return float("nan")

        raw = gym.make(ENV_ID, render_mode=None)
        wrapped = HillClimbRewardWrapper(raw, reward_fn=bad_reward)
        try:
            wrapped.reset(seed=42)
            with self.assertRaises((RuntimeError, ValueError)):
                wrapped.step(0)
        finally:
            wrapped.close()

    def test_classify_outcome_edge_cases(self):
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

        obs_stuck = np.array([250.0, 10.0, 0.0, 0.0, 0.1, 0.0, 1.0, 1.0], dtype=np.float32)
        succ_s, crash_s, tout_s, _ = classify_hill_climb_outcome(
            obs=obs_stuck,
            original_reward=-100.0,
            terminated=True,
            truncated=False,
        )
        self.assertFalse(succ_s)
        self.assertTrue(crash_s or tout_s)

        full_circle = float(2 * math.pi + 0.1)
        self.assertAlmostEqual(_wrap_angle(full_circle), 0.1, places=5)

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
            self.assertEqual(len(plt.get_fignums()), 0, "Leak di figure Matplotlib rilevato")

    def test_loaded_favorite_hybrid_access(self):
        fake_model = "fake_ppo_model"
        fake_meta = {"mission": "Test", "eval_metrics": {"eval_success_rate": 85.0}}
        fake_dir = Path("/fake/dir")
        fake_entry = {"id": 1, "nome": "Mio Test"}

        fav = LoadedFavorite(model=fake_model, metadata=fake_meta, dir_path=fake_dir, entry=fake_entry)

        m, meta = fav
        self.assertEqual(m, fake_model)
        self.assertEqual(meta["mission"], "Test")
        self.assertEqual(fav.model, fake_model)
        self.assertEqual(fav.metadata, fake_meta)
        self.assertEqual(fav["metadata"], fake_meta)
        self.assertEqual(fav.get("mission"), "Test")
        # Regressione B-04: get() deve restituire default su chiave assente
        self.assertEqual(fav.get("chiave_non_esistente", "fallback"), "fallback")

    # =========================================================================
    # REGRESSION TESTS — FISICA BOX2D & RENDERING (B-02, B-07..B-12, B-15)
    # =========================================================================

    def test_regression_b07_b08_physics_masses_suspensions_and_hitboxes(self):
        """Verifica rapporti di massa sani, sospensioni sotto Nyquist, maskBits rim_body e poligoni validi."""
        env = gym.make(ENV_ID, render_mode=None)
        try:
            env.reset(seed=42)
            car = env.unwrapped.agent.car

            # 1. rim_body non deve collidere con il terreno/erba (B-07)
            for wheel in car.wheels:
                rim_fix = wheel.rim_body.fixtures[0]
                self.assertEqual(
                    rim_fix.filterData.maskBits,
                    0,
                    "rim_body collide ancora con l'erba comportandosi come un freno bloccato",
                )
                # 2. Frequenza sospensione entro la zona di stabilità di Nyquist a 60 FPS (f <= 15 Hz)
                freq = getattr(wheel.dist_joint, "frequencyHz", getattr(wheel.dist_joint, "frequency", 8.5))
                self.assertLessEqual(freq, 15.0)
                self.assertLessEqual(wheel.dist_joint.dampingRatio, 2.0)

            # 3. Rapporto di massa telaio / testa entro soglia stabile per solver Gauss-Seidel (B-08)
            chassis_mass = car.chassis_body.mass
            head_mass = car.person.head.body.mass
            torso_mass = car.person.torso.body.mass
            self.assertGreater(head_mass, 0.0)
            self.assertLess(
                chassis_mass / head_mass,
                150.0,
                f"Rapporto massa telaio/testa patologico: {chassis_mass / head_mass:.1f}:1",
            )
            self.assertLess(chassis_mass / torso_mass, 150.0)

            # 4. Nessun vertice duplicato nel poligono principale del telaio (B-08)
            chassis_verts = [(round(v.x, 4), round(v.y, 4)) for v in car.shapes[0]]
            self.assertEqual(
                len(chassis_verts),
                len(set(chassis_verts)),
                f"Vertici duplicati rilevati nel telaio: {chassis_verts}",
            )
        finally:
            env.close()

    def test_regression_b09_contact_listener_anti_flickering_and_head_crash(self):
        """Verifica contatore intero on_ground anti-flickering e crash testa-terreno senza indici fragili."""
        env = gym.make(ENV_ID, render_mode=None)
        try:
            env.reset(seed=42)
            listener = ContactListener()
            agent = env.unwrapped.agent
            wheel = agent.car.wheels[0]
            head = agent.car.person.head
            ground = env.unwrapped.ground

            # Simula contatto sovrapposto tra due edge adiacenti: Begin(e1) -> Begin(e2) -> End(e1)
            c1 = SimpleNamespace(
                fixtureA=SimpleNamespace(body=wheel.body),
                fixtureB=SimpleNamespace(body=ground.grassBody),
            )
            c2 = SimpleNamespace(
                fixtureA=SimpleNamespace(body=ground.grassBody),
                fixtureB=SimpleNamespace(body=wheel.body),
            )

            wheel.ground_contacts = 0
            listener.BeginContact(c1)
            self.assertTrue(wheel.on_ground)
            listener.BeginContact(c2)
            listener.EndContact(c1)
            self.assertTrue(
                wheel.on_ground,
                "Flickering rilevato: on_ground è diventato False al cambio di edge adiacente",
            )
            listener.EndContact(c2)
            self.assertFalse(wheel.on_ground)

            # Verifica crash testa-terreno imposti sia agent.dead che car.dead
            head_contact = SimpleNamespace(
                fixtureA=SimpleNamespace(body=head.body),
                fixtureB=SimpleNamespace(body=ground.grassBody),
            )
            listener.BeginContact(head_contact)
            self.assertTrue(agent.dead)
            self.assertTrue(agent.car.dead)
        finally:
            env.close()

    def test_regression_b10_no_invisible_wall_body_leak_across_resets(self):
        """Verifica che _destroy_world() elimini tutti i corpi statici (incluso invisible_wall) ad ogni reset."""
        env = gym.make(ENV_ID, render_mode=None)
        try:
            env.reset(seed=10)
            initial_body_count = len(env.unwrapped.world.bodies)

            for s in (11, 12, 13, 14, 15):
                env.reset(seed=s)
                self.assertEqual(
                    len(env.unwrapped.world.bodies),
                    initial_body_count,
                    "Memory leak Box2D: il numero di corpi cresce tra i reset (probabile invisible_wall non distrutto)",
                )
        finally:
            env.close()

    def test_regression_b11_terrain_determinism_across_instances(self):
        """Verifica determinismo esatto della generazione Perlin a parità di seed su due istanze separate."""
        env_a = gym.make(ENV_ID, render_mode=None, original_noise=True)
        env_b = gym.make(ENV_ID, render_mode=None, original_noise=True)
        try:
            env_a.reset(seed=12345)
            env_b.reset(seed=12345)

            pts_a = np.array([(v.x, v.y) for v in env_a.unwrapped.ground.ground_vectors], dtype=np.float64)
            pts_b = np.array([(v.x, v.y) for v in env_b.unwrapped.ground.ground_vectors], dtype=np.float64)
            self.assertTrue(
                np.allclose(pts_a, pts_b, atol=1e-6),
                "Il terreno Perlin non è deterministico a parità di seed",
            )

            # Seed diversi devono produrre profili distinti
            env_b.reset(seed=99999)
            pts_c = np.array([(v.x, v.y) for v in env_b.unwrapped.ground.ground_vectors], dtype=np.float64)
            self.assertFalse(np.allclose(pts_a, pts_c, atol=1e-4))
        finally:
            env_a.close()
            env_b.close()

    def test_regression_b12_render_does_not_mutate_ground_vectors(self):
        """Verifica che draw_ground() non moltiplichi in-place ground_vectors per SCALE=30."""
        env = gym.make(ENV_ID, render_mode="rgb_array")
        try:
            env.reset(seed=42)
            before = [(v.x, v.y) for v in env.unwrapped.ground.ground_vectors[:15]]
            env.render()
            env.render()
            after = [(v.x, v.y) for v in env.unwrapped.ground.ground_vectors[:15]]

            for (x0, y0), (x1, y1) in zip(before, after):
                self.assertAlmostEqual(x0, x1, places=5, msg="ground_vectors.x mutato in-place da render()")
                self.assertAlmostEqual(y0, y1, places=5, msg="ground_vectors.y mutato in-place da render()")
        finally:
            env.close()

    # =========================================================================
    # REGRESSION TESTS — RL WRAPPER, VALUTAZIONE & API (B-01, B-04, B-05, B-06, B-14, B-16, B-17)
    # =========================================================================

    def test_regression_b01_b14_eval_duration_budget_and_empty_seeds(self):
        """Verifica budget temporale sufficiente per 800m e protezione da ZeroDivisionError su eval_seeds=[]."""
        sig = inspect.signature(evaluate_and_record)
        default_max_steps = sig.parameters["max_steps"].default
        default_frame_skip = sig.parameters["frame_skip"].default if "frame_skip" in sig.parameters else 1
        simulated_seconds = (default_max_steps * default_frame_skip) / float(hill_racing.FPS)
        self.assertGreaterEqual(
            simulated_seconds,
            50.0,
            f"Durata massima di valutazione insufficiente ({simulated_seconds:.1f}s) per raggiungere 800m",
        )

        res = evaluate_and_record(model=None, eval_seeds=[])
        self.assertEqual(res["total_episodes"], 0)
        self.assertEqual(res["success_rate"], 0.0)

    def test_regression_b06_wrapper_terminates_immediately_at_target_distance(self):
        """Verifica che il wrapper termini subito l'episodio al raggiungimento di target_distance."""
        raw = gym.make(ENV_ID, render_mode=None)
        wrapped = HillClimbRewardWrapper(raw, reward_fn=None, target_distance=5.0)
        try:
            wrapped.reset(seed=42)
            # Lo spawn è già a ~6.67m > target_distance=5.0m -> deve terminare al primo step
            obs, rew, term, trunc, info = wrapped.step(1)
            self.assertTrue(term, "L'episodio non è terminato al superamento di target_distance")
            self.assertTrue(info.get("is_success", False))
            self.assertIn("is_timeout", info)
        finally:
            wrapped.close()

    def test_regression_b17_student_reward_validation_and_readonly_obs(self):
        """Verifica accettazione scalari NumPy/Torch, rifiuto di None/tensori multi-dim e immutabilità obs."""
        raw = gym.make(ENV_ID, render_mode=None)
        try:
            # 1. Scalari NumPy e PyTorch validi
            for fn, expected in (
                (lambda obs, act: np.array([3.5], dtype=np.float32), 3.5),
                (lambda obs, act: torch.tensor([2.25], dtype=torch.float32), 2.25),
            ):
                w = HillClimbRewardWrapper(raw, reward_fn=fn)
                w.reset(seed=42)
                _, rew, _, _, _ = w.step(0)
                self.assertAlmostEqual(rew, expected, places=5)

            # 2. Tipi invalidi (None, tensore multi-elemento, array multi-elemento, +Inf)
            for bad_fn in (
                lambda obs, act: None,
                lambda obs, act: torch.tensor([1.0, 2.0]),
                lambda obs, act: np.array([1.0, 2.0]),
                lambda obs, act: float("inf"),
            ):
                w = HillClimbRewardWrapper(raw, reward_fn=bad_fn)
                w.reset(seed=42)
                with self.assertRaises((RuntimeError, ValueError)):
                    w.step(0)

            # 3. Mutazione in-place di obs dentro custom_reward deve fallire esplicitamente
            def mutating_reward(obs, act):
                obs[0] = 999.0
                return 1.0

            w_ro = HillClimbRewardWrapper(raw, reward_fn=mutating_reward)
            w_ro.reset(seed=42)
            with self.assertRaises((RuntimeError, ValueError)):
                w_ro.step(0)
        finally:
            raw.close()

    def test_regression_b04_b05_favorites_and_comprimi_archivio(self):
        """Verifica round-trip completo dei preferiti e creazione ZIP senza TypeError."""
        with tempfile.TemporaryDirectory() as td:
            tmp_runs = Path(td) / "runs"
            tmp_run_dir = tmp_runs / "run_hillclimb_20260101_120000"
            tmp_run_dir.mkdir(parents=True, exist_ok=True)
            save_run_metadata(
                tmp_run_dir,
                {
                    "timestamp": "2026-01-01 12:00:00",
                    "mission": {"key": "campagna_standard", "title": "Campagna Standard"},
                    "eval_metrics": {"eval_success_rate": 90.0, "mean_reward": 120.0, "max_distance": 845.0},
                },
            )

            tmp_fav_dir = tmp_runs / "favorites"
            tmp_mgr = BaseFavoritesManager(tmp_fav_dir)
            with patch.object(hc_favorites, "FAVORITES_DIR", tmp_fav_dir), patch.object(
                hc_favorites, "_mgr", tmp_mgr
            ):
                entry = hc_favorites.salva_nei_preferiti(tmp_run_dir, nome="Campione Test", note="Ottimo")
                self.assertEqual(entry["id"], 1)

                # Non deve sollevare AttributeError
                hc_favorites.mostra_preferiti()

                loaded = hc_favorites.carica_preferito(1)
                self.assertIsNotNone(loaded)
                self.assertEqual(loaded.entry["nome"], "Campione Test")
                self.assertEqual(loaded.get("chiave_assente", 42), 42)

            # Verifica comprimi_archivio (B-05)
            out_zip = Path(td) / "test_archivio.zip"
            zip_res = comprimi_archivio(source_dir=tmp_runs, output_zip=out_zip, scarica_su_colab=False)
            self.assertIsNotNone(zip_res)
            self.assertTrue(out_zip.exists())
            self.assertTrue(zipfile.is_zipfile(out_zip))

    def test_regression_b16_reporting_italian_accents_and_no_figure_leak(self):
        """Verifica generazione report MD/HTML/PDF con accenti italiani preservati e zero leak Matplotlib."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run_hillclimb_report"
            run_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "timestamp": "2026-01-01 12:00:00",
                "mission": "campagna_standard",
                "total_timesteps": 10_000,
                "eval_metrics": {
                    "success_rate": 85.0,
                    "crash_rate": 10.0,
                    "timeout_rate": 5.0,
                    "mean_distance": 760.0,
                    "max_distance": 880.0,
                    "mean_speed": 14.5,
                    "mean_airtime_pct": 12.0,
                },
                "history": [
                    {"seed": 42, "distance": 880.0, "max_speed": 14.5, "airtime_pct": 12.0, "reward": 210.0}
                ],
            }
            paths = genera_report_esperimento(
                run_dir=run_dir,
                metadata=metadata,
                titolo="Velocità e Stabilità in Salita",
                autore="Studente Niccolò",
                note="Verifica qualità accenti: à è é ì ò ù",
            )
            self.assertTrue(paths["md"].exists())
            self.assertTrue(paths["html"].exists())
            self.assertTrue(paths["pdf"].exists())
            self.assertIn("Velocità e Stabilità", paths["md"].read_text(encoding="utf-8"))
            self.assertIn("Niccolò", paths["html"].read_text(encoding="utf-8"))
            self.assertEqual(len(plt.get_fignums()), 0, "Leak di figure Matplotlib dopo genera_report_esperimento")


if __name__ == "__main__":
    unittest.main()
