"""
Modulo Agente RL per Play2Learn Hill Climb Racing.
Implementa il Feature Extractor a invarianza di traslazione e l'orchestrazione PPO.
"""

from __future__ import annotations

# Fix preventivo e definitivo per PyTorch / Sympy in Python 3.13 (Google Colab)
import sys
try:
    import sympy
    import sympy.logic
    import sympy.logic.boolalg
    sympy.logic = sympy.logic
    sympy.logic.boolalg = sympy.logic.boolalg
    sys.modules["sympy"].logic = sympy.logic
    sys.modules["sympy.logic"] = sympy.logic
    sys.modules["sympy.logic.boolalg"] = sympy.logic.boolalg
    import torch.utils._sympy.numbers
except Exception:
    pass

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
    Estrae feature invarianti per traslazione orizzontale pos_x dalle grandezze originali:
      - sin(angle) e cos(angle) (orientamento continuo)
      - rear_wheel_speed / 40.0 (velocità ruota posteriore normalizzata)
      - front_wheel_speed / 40.0 (velocità ruota anteriore normalizzata)
      - rear_wheel_ground (contatto ruota posteriore in {0, 1})
      - front_wheel_ground (contatto ruota anteriore in {0, 1})
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(6, features_dim),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[1] == 7:
            # 7D originale: [pos_x, pos_y, angle, r_spd, f_spd, rear_g, front_g]
            angle = observations[:, 2:3]
            sin_a = torch.sin(angle)
            cos_a = torch.cos(angle)
            r_spd = torch.clamp(observations[:, 3:4] / 40.0, -2.0, 2.0)
            f_spd = torch.clamp(observations[:, 4:5] / 40.0, -2.0, 2.0)
            contacts = observations[:, 5:7]
            feat = torch.cat([sin_a, cos_a, r_spd, f_spd, contacts], dim=1)
        else:
            # Fallback 8D: [pos_x, pos_y, vx, vy, angle, ang_vel, rear_g, front_g]
            angle = observations[:, 4:5]
            sin_a = torch.sin(angle)
            cos_a = torch.cos(angle)
            vx = torch.clamp(observations[:, 2:3] / 15.0, -2.0, 2.0)
            ang_vel = torch.clamp(observations[:, 5:6] / 10.0, -2.0, 2.0)
            contacts = observations[:, 6:8]
            feat = torch.cat([sin_a, cos_a, vx, ang_vel, contacts], dim=1)
        return self.net(feat)


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
            metric_name="distance",
        )

    def _extract_metric_from_info(self, info: dict[str, Any]) -> float:
        telem = info.get("telemetry", {})
        return float(telem.get("distance", telem.get("max_distance", info.get("score", 0.0))))


def create_ppo_agent(
    env: Any,
    learning_rate: float = DEFAULT_PPO_PARAMS["learning_rate"],
    total_timesteps: int = 300_000,
    seed: int = 42,
    device: str = "cpu",
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
