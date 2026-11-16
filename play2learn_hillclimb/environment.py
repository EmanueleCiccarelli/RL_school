"""
Modulo Ambiente e Wrapper di Reward per Play2Learn Hill Climb Racing.
Costruito direttamente sopra l'ambiente originale Farama Gymnasium `hill_racing_env`.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional
import numpy as np

import gymnasium as gym
from gymnasium import spaces

try:
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecEnv
    _SB3_AVAILABLE = True
except ImportError:
    Monitor = Any  # type: ignore
    DummyVecEnv = Any  # type: ignore
    SubprocVecEnv = Any  # type: ignore
    VecNormalize = Any  # type: ignore
    VecEnv = Any  # type: ignore
    _SB3_AVAILABLE = False

import hill_racing_env
from play2learn_core.base_env import BaseRewardWrapper
from play2learn_hillclimb.config import (
    ENV_ID,
    MISSIONS,
    TARGET_DISTANCE_SUCCESS,
    VECNORM_PARAMS,
)


def _wrap_angle(angle_rad: float) -> float:
    """Normalizza un angolo in radianti nell'intervallo [-pi, +pi]."""
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def classify_hill_climb_outcome(
    obs: np.ndarray,
    original_reward: float,
    terminated: bool,
    truncated: bool,
    unwrapped: Any = None,
) -> tuple[bool, bool, bool, bool]:
    """
    Unica fonte di verità per la classificazione dell'esito della corsa collinare.
    Restituisce: (is_success, is_crashed, is_timeout, is_flipped).

    Regole:
      - is_success: la vettura raggiunge o supera TARGET_DISTANCE_SUCCESS (800m).
      - is_crashed: la testa del pilota tocca terra (agente morto).
      - is_timeout: scadenza del tempo massimo o blocco in salita senza morte né successo.
      - is_flipped: inclinazione del telaio superiore a ~80° (angolo > 1.4 rad).
    """
    dist_x = float(obs[0]) if len(obs) > 0 else 0.0
    angle_rad = _wrap_angle(float(obs[4])) if len(obs) > 4 else 0.0
    is_flipped = bool(abs(angle_rad) > 1.4)

    # 1. Traguardo di successo raggiunto (priorità massima anche su timeout)
    if dist_x >= TARGET_DISTANCE_SUCCESS:
        return True, False, False, is_flipped

    # 2. Controllo morte fisica del pilota
    is_dead = False
    if unwrapped is not None:
        ag = getattr(unwrapped, "agent", None)
        if ag is not None:
            is_dead = bool(getattr(ag, "dead", False))
            car = getattr(ag, "car", None)
            if car is not None and getattr(car, "dead", False):
                is_dead = True
    if not is_dead and original_reward <= -90.0 and terminated:
        # Nell'engine originale alexzh3, morte = terminated con reward = -100
        is_dead = True

    if is_dead:
        return False, True, False, is_flipped

    # 3. Timeout / Blocco (step_stuck_counter scaduto o max_steps raggiunto)
    if truncated or terminated:
        return False, False, True, is_flipped

    return False, False, False, is_flipped


class HillClimbRewardWrapper(BaseRewardWrapper):
    """
    Wrapper didattico Play2Learn per Hill Climb Racing.
    Adatta lo spazio delle osservazioni Dict originale di alexzh3 in un vettore 8D continuo,
    facilita il calcolo della reward personalizzata dello studente e traccia la telemetria di bordo.
    """

    def __init__(
        self,
        env: gym.Env,
        reward_fn: Optional[Callable] = None,
        target_distance: float = TARGET_DISTANCE_SUCCESS,
    ):
        super().__init__(env, reward_fn=reward_fn)
        self.target_distance = float(target_distance)

        # Spazio di osservazione 8D didattico:
        # [dist_x, quota_y, vx, vy, angle_rad, ang_vel, rear_wheel_ground, front_wheel_ground]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -100.0, -50.0, -50.0, -math.pi, -20.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1200.0, 500.0, 50.0, 50.0, math.pi, 20.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._obs_buf = np.zeros(8, dtype=np.float32)
        self._max_distance_reached = 0.0
        self._max_speed_reached = 0.0
        self._airtime_steps = 0
        self._flip_count = 0
        self._total_steps = 0

    def _extract_8d_obs(self, raw_obs: Any) -> np.ndarray:
        """Estrae i valori fisici in un array numpy 8D continuo."""
        buf = self._obs_buf
        uw = self.env.unwrapped
        car = getattr(getattr(uw, "agent", None), "car", None)

        if car is not None and getattr(car, "chassis_body", None) is not None:
            cb = car.chassis_body
            pos_x = float(car.pos_x)
            pos_y = float(car.pos_y)
            vx = float(cb.linearVelocity.x)
            vy = float(cb.linearVelocity.y)
            ang = _wrap_angle(-cb.angle)
            v_ang = float(cb.angularVelocity)
            r_g = 1.0 if (len(car.wheels) > 0 and car.wheels[0].on_ground) else 0.0
            f_g = 1.0 if (len(car.wheels) > 1 and car.wheels[1].on_ground) else 0.0
        elif isinstance(raw_obs, dict):
            cp = raw_obs.get("chassis_position", [0.0, 0.0])
            pos_x = float(cp[0])
            pos_y = float(cp[1])
            vx, vy = 0.0, 0.0
            deg = float(raw_obs.get("chassis_angle", [0.0])[0])
            ang = _wrap_angle(-math.radians(deg))
            v_ang = 0.0
            og = raw_obs.get("on_ground", [0, 0])
            r_g = float(og[0]) if len(og) > 0 else 0.0
            f_g = float(og[1]) if len(og) > 1 else 0.0
        else:
            buf.fill(0.0)
            return buf.copy()

        buf[0] = pos_x
        buf[1] = pos_y
        buf[2] = vx
        buf[3] = vy
        buf[4] = ang
        buf[5] = v_ang
        buf[6] = r_g
        buf[7] = f_g
        return buf.copy()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        raw_obs, info = self.env.reset(seed=seed, options=options)
        self._max_distance_reached = 0.0
        self._max_speed_reached = 0.0
        self._airtime_steps = 0
        self._flip_count = 0
        self._total_steps = 0

        obs_8d = self._extract_8d_obs(raw_obs)
        self._max_distance_reached = float(obs_8d[0])
        return obs_8d, info

    def step(self, action: Any):
        raw_obs, orig_reward, term, trunc, info = self.env.step(action)
        obs_8d = self._extract_8d_obs(raw_obs)

        self._total_steps += 1
        dist_x = float(obs_8d[0])
        speed = math.sqrt(float(obs_8d[2]) ** 2 + float(obs_8d[3]) ** 2)
        if dist_x > self._max_distance_reached:
            self._max_distance_reached = dist_x
        if speed > self._max_speed_reached:
            self._max_speed_reached = speed
        if obs_8d[6] == 0.0 and obs_8d[7] == 0.0:
            self._airtime_steps += 1
        if abs(obs_8d[4]) > 1.4:
            self._flip_count += 1

        succ, crash, tout, flipped = classify_hill_climb_outcome(
            obs=obs_8d,
            original_reward=float(orig_reward),
            terminated=term,
            truncated=trunc,
            unwrapped=self.env.unwrapped,
        )

        if self.reward_fn is not None:
            raw_rew = self.call_student_reward(obs_8d, action, succ, crash, tout)
            reward = self.validate_reward_value(raw_rew)
        else:
            reward = float(orig_reward)

        if term or trunc:
            airtime_pct = (self._airtime_steps / max(1, self._total_steps)) * 100.0
            telem = {
                "distance": self._max_distance_reached,
                "max_distance": self._max_distance_reached,
                "max_speed": self._max_speed_reached,
                "airtime_pct": airtime_pct,
                "flips": self._flip_count,
                "steps": self._total_steps,
                "is_success": succ,
                "is_crashed": crash,
                "is_timeout": tout,
                "is_flipped": flipped,
            }
            info["telemetry"] = telem
            info["is_success"] = succ
            info["score"] = self._max_distance_reached

        return obs_8d, reward, term, trunc, info


def _make_single_env(
    mission: str = "campagna_standard",
    reward_fn: Optional[Callable] = None,
    render_mode: Optional[str] = None,
    target_distance: float = TARGET_DISTANCE_SUCCESS,
    seed: Optional[int] = None,
) -> gym.Env:
    """Crea una singola istanza configurata di HillRacingEnv con Monitor e Wrapper."""
    mission_cfg = MISSIONS.get(mission, MISSIONS["campagna_standard"])
    difficulty = float(mission_cfg.get("difficulty", -150.0))
    original_noise = bool(mission_cfg.get("original_noise", False))

    raw_env = gym.make(
        ENV_ID,
        render_mode=render_mode,
        action_space="discrete_3",
        reward_type="aggressive",
        difficulty=difficulty,
        original_noise=original_noise,
    )
    wrapped = HillClimbRewardWrapper(raw_env, reward_fn=reward_fn, target_distance=target_distance)
    monitored = Monitor(wrapped)
    return monitored


def make_parallel_envs(
    num_envs: int = 1,
    mission: str = "campagna_standard",
    reward_fn: Optional[Callable] = None,
    target_distance: float = TARGET_DISTANCE_SUCCESS,
    seed: int = 42,
) -> VecEnv:
    """Crea un ambiente vettoriale parallelizzato per Stable-Baselines3 PPO."""
    def _thunk(rank: int):
        def _init():
            return _make_single_env(
                mission=mission,
                reward_fn=reward_fn,
                render_mode=None,
                target_distance=target_distance,
                seed=seed + rank if seed is not None else None,
            )
        return _init

    env_fns = [_thunk(i) for i in range(num_envs)]
    if num_envs > 1:
        try:
            vec_env = SubprocVecEnv(env_fns)
        except Exception:
            vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    norm_env = VecNormalize(
        vec_env,
        norm_obs=VECNORM_PARAMS.get("norm_obs", False),
        norm_reward=VECNORM_PARAMS.get("norm_reward", True),
        clip_obs=VECNORM_PARAMS.get("clip_obs", 10.0),
        clip_reward=VECNORM_PARAMS.get("clip_reward", 10.0),
        gamma=VECNORM_PARAMS.get("gamma", 0.99),
    )
    return norm_env
