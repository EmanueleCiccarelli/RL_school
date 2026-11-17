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
    target_distance: float = TARGET_DISTANCE_SUCCESS,
) -> tuple[bool, bool, bool, bool]:
    """
    Unica fonte di verità per la classificazione dell'esito della corsa collinare.
    Restituisce: (is_success, is_crashed, is_timeout, is_flipped).

    Regole:
      - is_success: la vettura raggiunge o supera target_distance (default 800m).
      - is_crashed: la testa del pilota tocca terra (agente morto).
      - is_timeout: scadenza del tempo massimo o blocco in salita senza morte né successo.
      - is_flipped: inclinazione del telaio superiore a ~80° (angolo > 1.4 rad).
    """
    dist_x = float(obs[0]) if len(obs) > 0 else 0.0
    if len(obs) == 7:
        angle_rad = _wrap_angle(float(obs[2]))
    elif len(obs) > 4:
        angle_rad = _wrap_angle(float(obs[4]))
    else:
        angle_rad = 0.0
    is_flipped = bool(abs(angle_rad) > 1.4)

    # 1. Traguardo di successo raggiunto (priorità massima anche su timeout)
    if dist_x >= target_distance:
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
        is_dead = True

    if is_dead:
        return False, True, False, is_flipped

    # 3. Timeout / Blocco (step_stuck_counter scaduto o max_steps raggiunto)
    if truncated or terminated:
        return False, False, True, is_flipped

    return False, False, False, is_flipped


class ObservationArray(np.ndarray):
    """
    Array NumPy 1D contenente le sole 7 osservazioni fisiche originali di Hill Climb Racing,
    con supporto opzionale all'accesso per chiave come dizionario:
      - [0] pos_x: Distanza orizzontale lungo il tracciato (in metri)
      - [1] pos_y: Quota altimetrica verticale (in metri)
      - [2] angle: Angolo di inclinazione del telaio (in radianti, [-pi, +pi])
      - [3] rear_wheel_speed: Velocità di rotazione ruota posteriore (rad/s)
      - [4] front_wheel_speed: Velocità di rotazione ruota anteriore (rad/s)
      - [5] rear_on_ground: Contatto a terra ruota posteriore (1.0 = contatto, 0.0 = aria)
      - [6] front_on_ground: Contatto a terra ruota anteriore (1.0 = contatto, 0.0 = aria)
    """

    _KEY_MAP = {
        "chassis_position": slice(0, 2),
        "chassis_angle": slice(2, 3),
        "wheels_speed": slice(3, 5),
        "on_ground": slice(5, 7),
        "pos_x": 0,
        "dist_x": 0,
        "pos_y": 1,
        "quota_y": 1,
        "angle": 2,
        "rear_wheel_speed": 3,
        "front_wheel_speed": 4,
        "rear_on_ground": 5,
        "front_on_ground": 6,
        "rear_wheel": 5,
        "front_wheel": 6,
    }

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str) and item in self._KEY_MAP:
            return super().__getitem__(self._KEY_MAP[item])
        return super().__getitem__(item)


class HillClimbRewardWrapper(BaseRewardWrapper):
    """
    Wrapper didattico Play2Learn per Hill Climb Racing.
    Adatta lo spazio delle osservazioni Dict originale di alexzh3 in un vettore 7D continuo,
    restituendo solo ed esclusivamente le grandezze fisiche dell'ambiente originale:
      - [0] pos_x: Distanza orizzontale (in metri, spawn=6.67m, traguardo >= 800m)
      - [1] pos_y: Quota altimetrica verticale (in metri)
      - [2] angle: Inclinazione del telaio (radianti da -pi a +pi, 0=orizzontale)
      - [3] rear_wheel_speed: Velocità di rotazione ruota posteriore
      - [4] front_wheel_speed: Velocità di rotazione ruota anteriore
      - [5] rear_on_ground: Contatto gomma posteriore col terreno (1.0=aderenza, 0.0=aria)
      - [6] front_on_ground: Contatto gomma anteriore col terreno (1.0=aderenza, 0.0=aria)
    """

    def __init__(
        self,
        env: gym.Env,
        reward_fn: Optional[Callable] = None,
        target_distance: float = TARGET_DISTANCE_SUCCESS,
        frame_skip: int = 1,
    ):
        super().__init__(env, reward_fn=reward_fn)
        self.target_distance = float(target_distance)
        self.frame_skip = max(1, int(frame_skip))

        # Spazio di osservazione 7D didattico originale:
        # [dist_x, quota_y, angle, rear_wheel_speed, front_wheel_speed, rear_on_ground, front_on_ground]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -200.0, -math.pi, -60.0, -60.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1200.0, 800.0, math.pi, 60.0, 60.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._obs_buf = np.zeros(7, dtype=np.float32)
        self._max_distance_reached = 0.0
        self._max_speed_reached = 0.0
        self._airtime_steps = 0
        self._flip_count = 0
        self._total_steps = 0

    def _extract_obs(self, raw_obs: Any) -> ObservationArray:
        """Estrae i valori fisici originali in un array numpy 7D."""
        buf = self._obs_buf
        uw = self.env.unwrapped
        car = getattr(getattr(uw, "agent", None), "car", None)

        if car is not None and getattr(car, "chassis_body", None) is not None:
            cb = car.chassis_body
            pos_x = float(car.pos_x)
            pos_y = float(car.pos_y)
            ang = _wrap_angle(-cb.angle)
            r_spd = float(car.wheels[0].joint.speed) if len(car.wheels) > 0 and car.wheels[0].joint else 0.0
            f_spd = float(car.wheels[1].joint.speed) if len(car.wheels) > 1 and car.wheels[1].joint else 0.0
            r_g = 1.0 if (len(car.wheels) > 0 and car.wheels[0].on_ground) else 0.0
            f_g = 1.0 if (len(car.wheels) > 1 and car.wheels[1].on_ground) else 0.0
        elif isinstance(raw_obs, dict):
            cp = raw_obs.get("chassis_position", [0.0, 0.0])
            pos_x = float(cp[0])
            pos_y = float(cp[1])
            deg = float(raw_obs.get("chassis_angle", [0.0])[0])
            ang = _wrap_angle(-math.radians(deg))
            ws = raw_obs.get("wheels_speed", [0.0, 0.0])
            r_spd = float(ws[0]) if len(ws) > 0 else 0.0
            f_spd = float(ws[1]) if len(ws) > 1 else 0.0
            og = raw_obs.get("on_ground", [0, 0])
            r_g = float(og[0]) if len(og) > 0 else 0.0
            f_g = float(og[1]) if len(og) > 1 else 0.0
        else:
            buf.fill(0.0)
            return buf.copy().view(ObservationArray)

        buf[0] = pos_x
        buf[1] = pos_y
        buf[2] = ang
        buf[3] = r_spd
        buf[4] = f_spd
        buf[5] = r_g
        buf[6] = f_g
        return buf.copy().view(ObservationArray)

    def _extract_8d_obs(self, raw_obs: Any) -> np.ndarray:
        """Alias retrocompatibile per estrattore di osservazioni."""
        return self._extract_obs(raw_obs)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        raw_obs, info = self.env.reset(seed=seed, options=options)
        self._max_distance_reached = 0.0
        self._max_speed_reached = 0.0
        self._airtime_steps = 0
        self._flip_count = 0
        self._total_steps = 0

        obs_7d = self._extract_obs(raw_obs)
        self._max_distance_reached = float(obs_7d[0])
        return obs_7d, info

    def step(self, action: Any):
        total_orig_reward = 0.0
        term = False
        trunc = False
        info: dict[str, Any] = {}
        raw_obs = None

        for _ in range(self.frame_skip):
            raw_obs, orig_r, term, trunc, info = self.env.step(action)
            total_orig_reward += float(orig_r)
            if term or trunc:
                break

        obs_7d = self._extract_obs(raw_obs)
        self._total_steps += 1
        dist_x = float(obs_7d[0])
        wheel_speed = max(abs(float(obs_7d[3])), abs(float(obs_7d[4])))
        if dist_x > self._max_distance_reached:
            self._max_distance_reached = dist_x
        if wheel_speed > self._max_speed_reached:
            self._max_speed_reached = wheel_speed
        if obs_7d[5] == 0.0 and obs_7d[6] == 0.0:
            self._airtime_steps += 1
        if abs(obs_7d[2]) > 1.4:
            self._flip_count += 1

        succ, crash, tout, flipped = classify_hill_climb_outcome(
            obs=obs_7d,
            original_reward=total_orig_reward,
            terminated=term,
            truncated=trunc,
            unwrapped=self.env.unwrapped,
            target_distance=self.target_distance,
        )

        # Termina immediatamente al traguardo per evitare reward hacking
        if succ:
            term = True

        if self.reward_fn is not None:
            obs_ro = obs_7d.copy().view(ObservationArray)
            obs_ro.setflags(write=False)
            reward = self.call_student_reward(obs_ro, action, succ, crash, tout)
        else:
            reward = total_orig_reward

        info["is_success"] = succ
        info["is_timeout"] = tout
        info["score"] = self._max_distance_reached

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

        return obs_7d, reward, term, trunc, info


def _make_single_env(
    mission: str = "campagna_standard",
    reward_fn: Optional[Callable] = None,
    render_mode: Optional[str] = None,
    target_distance: float = TARGET_DISTANCE_SUCCESS,
    frame_skip: int = 1,
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
    wrapped = HillClimbRewardWrapper(
        raw_env,
        reward_fn=reward_fn,
        target_distance=target_distance,
        frame_skip=frame_skip,
    )
    monitored = Monitor(wrapped)
    return monitored


def make_parallel_envs(
    num_envs: int = 1,
    mission: str = "campagna_standard",
    reward_fn: Optional[Callable] = None,
    target_distance: float = TARGET_DISTANCE_SUCCESS,
    frame_skip: int = 1,
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
                frame_skip=frame_skip,
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
