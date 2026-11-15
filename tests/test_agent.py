import unittest
from play2learn_lunarlander.agent import RewardTrackerCallback


class TestAgent(unittest.TestCase):
    def test_tracker_sliding_window(self):
        tracker = RewardTrackerCallback(
            total_timesteps=1000,
            show_progress=False,
            window_size=5,
        )

        class FakeInfo:
            def __init__(self, r, l, succ, fuel):
                self.d = {
                    "episode": {"r": r, "l": l},
                    "is_success": succ,
                    "is_crashed": not succ,
                    "is_timeout": False,
                    "telemetry": {
                        "fuel_used": fuel,
                        "air_steps": l,
                        "ground_steps": 0,
                        "final_abs_x": 0.05,
                        "final_abs_vy": 0.1,
                        "final_abs_angle": 0.02,
                    },
                }

            def get(self, k, default=None):
                return self.d.get(k, default)

        tracker.locals = {
            "infos": [
                FakeInfo(r=100.0, l=150, succ=True, fuel=40.0),
                FakeInfo(r=50.0, l=120, succ=False, fuel=30.0),
                FakeInfo(r=150.0, l=160, succ=True, fuel=50.0),
                FakeInfo(r=200.0, l=140, succ=True, fuel=45.0),
                FakeInfo(r=0.0, l=80, succ=False, fuel=20.0),
            ]
        }

        tracker._on_step()

        self.assertEqual(len(tracker.episode_rewards), 5)
        self.assertEqual(tracker.successes, 3)
        self.assertEqual(tracker.crashes, 2)
        self.assertEqual(tracker.current_moving_success_rate, 60.0)
        self.assertEqual(tracker.current_moving_reward, 100.0)
        self.assertEqual(tracker.current_moving_fuel, 37.0)


if __name__ == "__main__":
    unittest.main()
