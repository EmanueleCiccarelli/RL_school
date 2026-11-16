"""
Modulo Addestramento Agenti PPO per Hill Climb Racing (Play2Learn).
Include feature extraction invariante alla traslazione, rollout auto-bilanciato
e callback O(1) con snapshot dei pesi in RAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecNormalize

from play2learn_core.agent import BaseRewardTrackerCallback, LinearSchedule
from play2learn_hillclimb.config import DEFAULT_PPO_PARAMS

TARGET_ROLLOUT_SIZE = 2048


class HillClimbFeaturesExtractor(BaseFeaturesExtractor):
    """
    Trasforma lo stato raw 8D [x, y, vx, vy, angle, vangle, rear, front]
    in feature invarianti alla traslazione e scalate in [-1, 1] per MlpPolicy,
    mantenendo norm_obs=False e piena compatibilita con PPO.load() e custom_reward.
    """

    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=7)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        vx = torch.clamp(observations[:, 2:3] / 15.0, -3.0, 3.0)
        vy = torch.clamp(observations[:, 3:4] / 15.0, -3.0, 3.0)
        angle = observations[:, 4:5]
        sin_a = torch.sin(angle)
        cos_a = torch.cos(angle)
        vangle = torch.clamp(observations[:, 5:6] / 10.0, -3.0, 3.0)
        rear = observations[:, 6:7]
        front = observations[:, 7:8]
        return torch.cat([vx, vy, sin_a, cos_a, vangle, rear, front], dim=1)


class HillClimbRewardTrackerCallback(BaseRewardTrackerCallback):
    """Callback didattica O(1) per monitorare distanza percorsa e stabilita."""

    def __init__(
        self,
        total_timesteps: int,
        show_progress: bool = True,
        save_checkpoints: bool = True,
        checkpoint_dir: Path | str | None = None,
        save_freq: int = 10_000,
        window_size: int = 50,
        milestone_callback: Callable[[int, Any], None] | None = None,
        milestone_freq: int = 30_000,
        verbose: int = 0,
    ):
        super().__init__(
            total_timesteps=total_timesteps,
            show_progress=show_progress,
            save_checkpoints=save_checkpoints,
            checkpoint_dir=checkpoint_dir,
            save_freq=save_freq,
            window_size=window_size,
            milestone_callback=milestone_callback,
            milestone_freq=milestone_freq,
            verbose=verbose,
            metric_name="max_distance",
        )


def build_hill_agent(
    env: VecNormalize,
    learning_rate: float = 3e-4,
    ppo_kwargs: dict[str, Any] | None = None,
    seed: int = 42,
) -> PPO:
    """Costruisce un modello PPO calibrato con feature scaling invariante e rollout bilanciato."""
    params = dict(DEFAULT_PPO_PARAMS)
    params["learning_rate"] = learning_rate
    if ppo_kwargs:
        params.update(ppo_kwargs)

    if not (ppo_kwargs and "n_steps" in ppo_kwargs):
        n_envs = max(1, getattr(env, "num_envs", 4))
        batch_size = int(params.get("batch_size", 64))
        raw_steps = max(batch_size, TARGET_ROLLOUT_SIZE // n_envs)
        params["n_steps"] = max(batch_size, (raw_steps // batch_size) * batch_size)

    lr_val = params.pop("learning_rate")
    lr_schedule = lr_val if callable(lr_val) else LinearSchedule(initial_value=float(lr_val), min_value=1e-5)

    policy_kwargs = dict(params.pop("policy_kwargs", {}))
    policy_kwargs.setdefault("features_extractor_class", HillClimbFeaturesExtractor)
    policy_kwargs.setdefault("net_arch", dict(pi=[64, 64], vf=[64, 64]))
    policy_kwargs.setdefault("ortho_init", True)

    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr_schedule,
        policy_kwargs=policy_kwargs,
        seed=seed,
        device="cpu",
        verbose=0,
        **params,
    )
