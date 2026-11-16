"""
Wrapper Didattico e Classificatore Esiti per Hill Climb Racing (Play2Learn).
Isola PPO da eccezioni dello studente, controlla valori non finiti (NaN/Inf)
e produce una ricca telemetria ad alte prestazioni con fast-path non terminale.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from play2learn_core.base_env import BaseRewardWrapper
from play2learn_hillclimb.config import TARGET_DISTANCE_SUCCESS


def _wrap_angle(rad: float) -> float:
    """Riporta l'angolo in radianti nell'intervallo [-pi, pi]."""
    return math.atan2(math.sin(rad), math.cos(rad))


def classify_hill_climb_outcome(
    obs: Any,
    original_reward: float,
    terminated: bool,
    truncated: bool,
    unwrapped: Any = None,
) -> tuple[bool, bool, bool, bool]:
    """
    Sorgente unica di verità per l'esito di guida in Hill Climb Racing.
    Restituisce: (is_success, is_crashed, is_timeout, is_flipped).
    """
    if not terminated and not truncated:
        return False, False, False, False

    agent_obj = getattr(unwrapped, "agent", None)
    is_dead = bool(getattr(agent_obj, "dead", False))
    score = float(getattr(agent_obj, "score", 0.0))

    try:
        x_dist = float(obs.item(0))
        angle_rad = abs(_wrap_angle(float(obs.item(4))))
    except (AttributeError, ValueError):
        x_dist = float(obs[0])
        angle_rad = abs(_wrap_angle(float(obs[4])))

    is_flipped = is_dead or (angle_rad > 1.4)

    # 1. Impatto testa / morte ha sempre precedenza
    if is_dead:
        return False, True, False, is_flipped

    # 2. Distanza obiettivo raggiunta (anche se scade max_steps senza schianto)
    if score >= TARGET_DISTANCE_SUCCESS or x_dist >= TARGET_DISTANCE_SUCCESS:
        return True, False, False, False

    # 3. Timeout o blocco in salita (step_stuck_counter termina con is_dead=False)
    is_stuck = bool(terminated and not is_dead and not is_flipped)
    if truncated or is_stuck:
        return False, False, True, is_flipped

    return False, True, False, is_flipped


class HillClimbRewardWrapper(BaseRewardWrapper):
    """Wrapper didattico ad alte prestazioni con fast-path non terminale e guardia NaN/Inf."""

    __slots__ = (
        "_unwrapped",
        "_isfinite",
        "episode_steps",
        "cumulative_distance",
        "airtime_steps",
        "max_speed_reached",
    )

    def __init__(self, env: gym.Env, reward_fn: Callable | None = None):
        super().__init__(env, reward_fn=reward_fn)
        self._unwrapped = getattr(env, "unwrapped", env)
        self._isfinite = math.isfinite
        self.episode_steps: int = 0
        self.cumulative_distance: float = 0.0
        self.airtime_steps: int = 0
        self.max_speed_reached: float = 0.0

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        self.episode_steps = 0
        self.cumulative_distance = 0.0
        self.airtime_steps = 0
        self.max_speed_reached = 0.0
        return super().reset(**kwargs)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        act_int = int(action)
        obs, original_reward, terminated, truncated, info = self.env.step(act_int)
        self.episode_steps += 1

        # Guardia NaN/Inf rapida tramite riduzione scalare float (zero allocazioni bool array)
        if not self._isfinite(float(obs.sum())):
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        x_dist = float(obs[0])
        abs_vx = abs(float(obs[2]))
        if x_dist > self.cumulative_distance:
            self.cumulative_distance = x_dist
        if abs_vx > self.max_speed_reached:
            self.max_speed_reached = abs_vx
        if obs[6] <= 0.5 and obs[7] <= 0.5:
            self.airtime_steps += 1

        # Fast-path (>99.5% step): zero chiamate a classify_outcome, zero allocazioni dict telemetry
        if not (terminated or truncated):
            caller = self._caller
            if caller is not None:
                try:
                    reward = float(caller(obs, act_int, False, False, False))
                    if not self._isfinite(reward):
                        raise ValueError(f"Valore non finito restituito: {reward}")
                except Exception as exc:
                    raise RuntimeError(
                        f"[Play2Learn Error] Errore nella tua 'custom_reward': {exc}"
                    ) from exc
            else:
                reward = float(original_reward)
                if not self._isfinite(reward):
                    reward = 0.0
            return obs, reward, False, False, info

        is_success, is_crashed, is_timeout, is_flipped = classify_hill_climb_outcome(
            obs=obs,
            original_reward=float(original_reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            unwrapped=self._unwrapped,
        )

        if self._caller is not None:
            reward = self.call_student_reward(obs, act_int, is_success, is_crashed, is_timeout)
        else:
            reward = self.validate_reward_value(original_reward)

        info["is_success"] = is_success
        info["is_crashed"] = is_crashed
        info["is_timeout"] = is_timeout
        info["is_flipped"] = is_flipped
        info["telemetry"] = {
            "max_distance": self.cumulative_distance,
            "airtime_steps": self.airtime_steps,
            "max_speed": self.max_speed_reached,
            "flight_steps": self.episode_steps,
        }
        return obs, reward, terminated, truncated, info
