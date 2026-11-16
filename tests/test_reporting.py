import unittest
import math
import numpy as np
from pathlib import Path

from play2learn_lunarlander.reporting import (
    _serialize_eval_metrics,
    _load_reward_fn_from_source,
    genera_report_esperimento,
)


class TestReporting(unittest.TestCase):
    def test_serialize_eval_metrics_strips_paths(self):
        raw_stats = {
            "eval_success_rate": 80.0,
            "mean_reward": 150.5,
            "mean_fuel": 42.0,
            "videos": [Path("a/b/c.mp4"), Path("x/y/z.mp4")],
            "gif_path": Path("runs/milestones.gif"),
            "nested": {"val": 10, "path": Path("dummy")},
        }
        serialized = _serialize_eval_metrics(raw_stats)
        self.assertIn("eval_success_rate", serialized)
        self.assertEqual(serialized["eval_success_rate"], 80.0)
        self.assertNotIn("videos", serialized)
        self.assertNotIn("gif_path", serialized)
        self.assertNotIn("nested", serialized)

    def test_load_reward_fn_with_numpy_and_math(self):
        source = """
def custom_reward(obs, action, is_success, is_crashed, is_timeout):
    dist = np.linalg.norm(obs[:2])
    decay = math.exp(-dist)
    return float(decay)
"""
        fn = _load_reward_fn_from_source(source)
        self.assertIsNotNone(fn)
        obs = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        val = fn(obs, 0, False, False, False)
        self.assertIsInstance(val, float)
        self.assertGreater(val, 0.0)

    def test_html_escaping_in_report(self):
        import tempfile
        import shutil
        import json

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            meta = {
                "mission": {"title": "Test <Danger>"},
                "reward_source_code": "if x < 0.5 and y > 0.2:\n    return True <script>alert(1)</script>",
                "eval_metrics": {"eval_success_rate": 80.0, "mean_fuel": 40.0},
                "notes": "Note con <bold>tag</bold>",
            }
            (tmp_dir / "run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")

            res = genera_report_esperimento(
                run_or_fav=tmp_dir,
                autore="Studente <script>",
                titolo="Titolo <Escape>",
            )
            html_content = res["html_path"].read_text(encoding="utf-8")
            self.assertNotIn("<script>", html_content)
            self.assertIn("&lt;script&gt;", html_content)
            self.assertIn("&lt;Danger&gt;", html_content)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_labresult_mostra_evoluzione(self):
        import tempfile
        import shutil
        from play2learn_lunarlander.lab import LabResult, mostra_evoluzione

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            fake_mp4 = tmp_dir / "evoluzione_apprendimento.mp4"
            fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")
            lr = LabResult({
                "checkpoint_dir": tmp_dir,
                "gif_evoluzione_path": fake_mp4,
            })
            # Visualizzazione con media esistente
            ret = lr.mostra_evoluzione()
            self.assertIsNotNone(ret)
            self.assertEqual(lr.genera_evoluzione(), ret)

            # Chiamata tramite funzione top-level
            ret_top = mostra_evoluzione(lr)
            self.assertIsNotNone(ret_top)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
