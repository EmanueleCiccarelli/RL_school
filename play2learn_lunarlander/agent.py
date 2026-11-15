"""
Modulo per la gestione dell'agente RL (PPO) e dei callback di telemetria e checkpointing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecEnv
except ImportError:
    # Fallback sicuro per ambienti in cui le librerie non sono ancora installate
    PPO = Any  # type: ignore
    BaseCallback = object  # type: ignore
    VecEnv = Any  # type: ignore

from play2learn_lunarlander.config import DEFAULT_PPO_PARAMS


class RewardTrackerCallback(BaseCallback):
    """
    Callback di monitoraggio in tempo reale.
    Raccoglie metriche telemetriche per ogni episodio completato:
    ricompensa cumulata, durata dello step, esito (successo/schianto/timeout),
    carburante consumato, velocità di impatto e vicinanza alla piazzola.
    Supporta inoltre il salvataggio automatico del 'miglior modello' (best checkpoint).
    """

    def __init__(self, best_model_save_dir: Path | str | None = None, verbose: int = 0):
        super().__init__(verbose)
        self.best_model_save_dir = Path(best_model_save_dir) if best_model_save_dir else None

        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.outcomes: list[str] = []
        self.fuel_used_list: list[float] = []
        self.air_steps_list: list[int] = []
        self.ground_steps_list: list[int] = []
        self.final_x_list: list[float] = []
        self.final_vy_list: list[float] = []
        self.final_angle_list: list[float] = []

        self.successes: int = 0
        self.crashes: int = 0
        self.timeouts: int = 0

        self.best_moving_success_rate: float = -1.0
        self.best_model_path: Path | None = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_lengths.append(int(info["episode"]["l"]))

                if info.get("is_success", False):
                    self.outcomes.append("success")
                    self.successes += 1
                elif info.get("is_crashed", False):
                    self.outcomes.append("crash")
                    self.crashes += 1
                elif info.get("is_timeout", False):
                    self.outcomes.append("timeout")
                    self.timeouts += 1
                else:
                    self.outcomes.append("unknown")

                tel = info.get("telemetry", {})
                self.fuel_used_list.append(float(tel.get("fuel_used", 0.0)))
                self.air_steps_list.append(int(tel.get("air_steps", info["episode"]["l"])))
                self.ground_steps_list.append(int(tel.get("ground_steps", 0)))
                self.final_x_list.append(float(tel.get("final_abs_x", 0.0)))
                self.final_vy_list.append(float(tel.get("final_abs_vy", 0.0)))
                self.final_angle_list.append(float(tel.get("final_abs_angle", 0.0)))

                # Checkpoint automatico del miglior modello se la media mobile del successo supera il record
                if self.best_model_save_dir and len(self.outcomes) >= 30:
                    window = 30
                    recent = self.outcomes[-window:]
                    curr_rate = recent.count("success") / window * 100.0
                    if curr_rate > self.best_moving_success_rate and curr_rate >= 40.0:
                        self.best_moving_success_rate = curr_rate
                        self.best_model_save_dir.mkdir(parents=True, exist_ok=True)
                        save_p = self.best_model_save_dir / "best_model.zip"
                        self.model.save(str(save_p))
                        self.best_model_path = save_p

        return True


def create_ppo_agent(
    env: VecEnv,
    seed: int = 42,
    device: str = "cpu",
    custom_hyperparams: dict[str, Any] | None = None,
    verbose: int = 1,
) -> PPO:
    """
    Istanzia un agente PPO pre-configurato con iperparametri ottimali per Colab.
    """
    params = dict(DEFAULT_PPO_PARAMS)
    if custom_hyperparams:
        params.update(custom_hyperparams)

    return PPO(
        env=env,
        seed=seed,
        device=device,
        verbose=verbose,
        **params,
    )
