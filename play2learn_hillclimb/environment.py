"""
Wrapper Didattico e Classificatore Esiti per Hill Climb Racing (Play2Learn).
Isola PPO da eccezioni dello studente, controlla valori non finiti (NaN/Inf)
e produce una ricca telemetria ad ogni passo di simulazione.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from play2learn_core.base_env import BaseRewardWrapper
from play2learn_hillclimb.config import TARGET_DISTANCE_SUCCESS


def classify_hill_climb_outcome(
    obs: Any,
    original_reward: float,
    terminated: bool,
    truncated: bool,
    unwrapped: Any = None,
) -> tuple[bool, bool, bool, bool]:
    """
    Sorgente unica di verità per la classificazione dell'esito dell'episodio di guida.
    Restituisce: (is_success, is_crashed, is_timeout, is_flipped).
    """
    is_timeout = bool(truncated and not terminated)
    if not terminated and not truncated:
        return False, False, False, False

    agent_obj = getattr(unwrapped, "agent", None)
    is_dead = bool(getattr(agent_obj, "dead", False))
    score = float(getattr(agent_obj, "score", 0.0))

    try:
        x_dist = float(obs.item(0))
        angle_rad = abs(float(obs.item(4)))
    except (AttributeError, ValueError):
        x_dist = float(obs[0])
        angle_rad = abs(float(obs[4]))

    is_flipped = is_dead or (angle_rad > 1.4)  # ~80 gradi di inclinazione estrema

    if is_dead:
        return False, True, False, is_flipped

    if is_timeout:
        return False, False, True, is_flipped

    # Traguardo di successo: raggiungimento della distanza obiettivo (>= 800m)
    if score >= TARGET_DISTANCE_SUCCESS or x_dist >= TARGET_DISTANCE_SUCCESS:
        return True, False, False, False

    return False, True, False, is_flipped


class HillClimbRewardWrapper(BaseRewardWrapper):
    """
    Wrapper didattico specializzato per Hill Climb Racing.
    Permette agli studenti di definire funzioni di reward per premiare la velocità in salita,
    l'airtime controllato, il contatto con il suolo e penalizzare il ribaltamento.
    """

    def __init__(self, env: gym.Env, reward_fn: Callable | None = None):
        super().__init__(env, reward_fn=reward_fn)
        self.episode_steps = 0
        self.cumulative_distance = 0.0
        self.airtime_steps = 0
        self.max_speed_reached = 0.0

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        self.episode_steps = 0
        self.cumulative_distance = 0.0
        self.airtime_steps = 0
        self.max_speed_reached = 0.0
        return super().reset(**kwargs)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1

        unwrapped = getattr(self.env, "unwrapped", self.env)
        is_success, is_crashed, is_timeout, is_flipped = classify_hill_climb_outcome(
            obs=obs,
            original_reward=original_reward,
            terminated=terminated,
            truncated=truncated,
            unwrapped=unwrapped,
        )

        # Aggiornamento telemetria
        x_dist = float(obs[0]) if hasattr(obs, "__getitem__") else 0.0
        vx = float(obs[2]) if hasattr(obs, "__getitem__") else 0.0
        self.cumulative_distance = max(self.cumulative_distance, x_dist)
        self.max_speed_reached = max(self.max_speed_reached, abs(vx))

        rear_on = bool(obs[6] > 0.5) if hasattr(obs, "__getitem__") else False
        front_on = bool(obs[7] > 0.5) if hasattr(obs, "__getitem__") else False
        if not rear_on and not front_on:
            self.airtime_steps += 1

        # Calcolo reward dello studente o default
        if self.reward_fn is not None:
            reward = self.call_student_reward(
                obs=obs,
                action=action,
                is_success=is_success,
                is_crashed=is_crashed,
                is_timeout=is_timeout,
            )
        else:
            reward = float(original_reward)

        info.update({
            "is_success": is_success,
            "is_crashed": is_crashed,
            "is_timeout": is_timeout,
            "is_flipped": is_flipped,
            "telemetry": {
                "max_distance": self.cumulative_distance,
                "airtime_steps": self.airtime_steps,
                "max_speed": self.max_speed_reached,
                "flight_steps": self.episode_steps,
            },
        })

        return obs, reward, terminated, truncated, info
