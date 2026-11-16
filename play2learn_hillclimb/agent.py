"""
Modulo Agente RL per Play2Learn Hill Climb Racing.
Implementa il Feature Extractor a invarianza di traslazione e l'orchestrazione PPO.
"""

from __future__ import annotations

import math
from typing import Any, Callable
import torch
import torch.nn as nn
import gymnasium as gym

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    _SB3_AVAILABLE = True
except ImportError:
    PPO = Any  # type: ignore
    BaseFeaturesExtractor = object  # type: ignore
    _SB3_AVAILABLE = False

from play2learn_core.agent import (
    create_linear_schedule,
    BaseRewardTrackerCallback,
    attach_jit_rollout_forward,
)
from play2learn_hillclimb.config import DEFAULT_PPO_PARAMS


class HillClimbFeaturesExtractor(BaseFeaturesExtractor):
    """
    Estrattore di feature specializzato per Hill Climb Racing.
    Elimina le coordinate spaziali assolute non stazionarie (x in [0, 1000] e y in [0, 700])
    trasformando lo stato 8D in un vettore a 7 dimensioni invariante per traslazione
    e opportunamente scalato per le attivazioni delle reti neurali:
      - vx / 15.0 (velocità orizzontale normalizzata)
      - vy / 15.0 (velocità verticale normalizzata)
      - sin(angle) e cos(angle) (orientamento trigonometrico continuo)
      - ang_vel / 10.0 (velocità angolare normalizzata)
      - rear_wheel_ground (contatto ruota posteriore in {0, 1})
      - front_wheel_ground (contatto ruota anteriore in {0, 1})
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(7, features_dim),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (batch_size, 8)
        # [x, y, vx, vy, angle, ang_vel, rear_g, front_g]
        vx = observations[:, 2:3] / 15.0
        vy = observations[:, 3:4] / 15.0
        angle = observations[:, 4:5]
        sin_a = torch.sin(angle)
        cos_a = torch.cos(angle)
        ang_vel = observations[:, 5:6] / 10.0
        contacts = observations[:, 6:8]

        feat_7d = torch.cat([vx, vy, sin_a, cos_a, ang_vel, contacts], dim=1)
        return self.net(feat_7d)


class HillClimbRewardTrackerCallback(BaseRewardTrackerCallback):
    """
    Callback didattico Play2Learn per Hill Climb Racing.
    Mostra una barra di avanzamento fluida con reward mobile, percentuale di successi
    e distanza massima raggiunta.
    """

    def __init__(
        self,
        total_timesteps: int,
        window_size: int = 20,
        show_progress: bool = True,
        save_checkpoints: bool = False,
        checkpoint_dir: Any = None,
        save_freq: int = 10_000,
        milestone_callback: Any = None,
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
        )

    def _extract_metric_from_info(self, info: dict[str, Any]) -> float:
        telem = info.get("telemetry", {})
        return float(telem.get("distance", telem.get("max_distance", info.get("score", 0.0))))


def create_ppo_agent(
    env: Any,
    learning_rate: float = DEFAULT_PPO_PARAMS["learning_rate"],
    total_timesteps: int = 300_000,
    seed: int = 42,
    device: str = "auto",
) -> PPO:
    """Crea e configura l'agente PPO con policy invariante per traslazione."""
    lr_schedule = create_linear_schedule(learning_rate, min_value=1e-5)

    policy_kwargs = {
        "features_extractor_class": HillClimbFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 64},
        "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        "activation_fn": nn.Tanh,
    }

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr_schedule,
        n_steps=DEFAULT_PPO_PARAMS["n_steps"],
        batch_size=DEFAULT_PPO_PARAMS["batch_size"],
        n_epochs=DEFAULT_PPO_PARAMS["n_epochs"],
        gamma=DEFAULT_PPO_PARAMS["gamma"],
        gae_lambda=DEFAULT_PPO_PARAMS["gae_lambda"],
        clip_range=DEFAULT_PPO_PARAMS["clip_range"],
        ent_coef=DEFAULT_PPO_PARAMS["ent_coef"],
        vf_coef=DEFAULT_PPO_PARAMS["vf_coef"],
        max_grad_norm=DEFAULT_PPO_PARAMS["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
        device=device,
    )
    return model
