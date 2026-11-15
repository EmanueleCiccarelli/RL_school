import unittest
import tempfile
import json
import shutil
from pathlib import Path

from play2learn_lunarlander.storage import (
    create_run_directory,
    save_run_metadata,
    find_run_by_timestamp,
)


class TestStorage(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_create_and_find_run_dir(self):
        run_dir = create_run_directory(mission="test_mission", base_dir=self.test_dir)
        self.assertTrue(run_dir.exists())
        self.assertTrue((run_dir / "videos").exists())

        # Crea un metadato minimo per rendere la run indicizzabile
        (run_dir / "run_metadata.json").write_text("{}", encoding="utf-8")

        found = find_run_by_timestamp("latest", runs_dir=self.test_dir)
        self.assertEqual(found, run_dir)

    def test_save_metadata(self):
        run_dir = create_run_directory(mission="luna_standard", base_dir=self.test_dir)

        def dummy_reward(obs, action):
            return 42.0

        meta_path = save_run_metadata(
            run_dir=run_dir,
            mission="luna_standard",
            mission_cfg={"title": "Test Mission"},
            env_kwargs={"gravity": -10.0},
            total_timesteps=1000,
            elapsed_seconds=1.5,
            reward_fn=dummy_reward,
            eval_stats={"eval_success_rate": 85.0, "mean_fuel": 45.0},
            diagnostics={"formatted_text": "Ottimo volo"},
        )

        self.assertTrue(meta_path.exists())
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(data["mission"]["title"], "Test Mission")
        self.assertEqual(data["eval_metrics"]["eval_success_rate"], 85.0)
        self.assertIn("dummy_reward", data["reward_source_code"])


if __name__ == "__main__":
    unittest.main()
