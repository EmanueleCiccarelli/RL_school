"""
Modulo Addestramento Agenti PPO per Hill Climb Racing (Play2Learn).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from play2learn_core.agent import BaseRewardTrackerCallback, create_linear_schedule, make_parallel_envs
from play2learn_hillclimb.config import DEFAULT_PPO_PARAMS


class HillClimbRewardTrackerCallback(BaseRewardTrackerCallback):
    """Callback didattica ad alte prestazioni per monitorare la distanza percorsa e la stabilità."""

    def __init__(
        self,
        total_timesteps: int,
        show_progress: bool = True,
        save_checkpoints: bool = True,
        checkpoint_dir: Path | str | None = None,
        save_freq: int = 10_000,
        window_size: int = 100,
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
    """Costruisce un modello PPO calibrato per la fisica di Hill Climb Racing."""
    params = dict(DEFAULT_PPO_PARAMS)
    if ppo_kwargs:
        params.update(ppo_kwargs)

    lr_schedule = create_linear_schedule(initial_value=learning_rate, min_value=1e-5)

    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr_schedule,
        n_steps=params["n_steps"],
        batch_size=params["batch_size"],
        n_epochs=params["n_epochs"],
        gamma=params["gamma"],
        gae_lambda=params["gae_lambda"],
        clip_range=params["clip_range"],
        ent_coef=params["ent_coef"],
        max_grad_norm=params["max_grad_norm"],
        target_kl=params["target_kl"],
        seed=seed,
        verbose=0,
    )
