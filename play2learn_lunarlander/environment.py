"""
play2learn_lunarlander/environment.py (Optimized Architecture)
- _NullParticle: elimina allocazioni Box2D di particelle fumo in modalità headless (0 drift)
- _FastWorldProxy: riduce iterazioni solver Box2D (180,60)->(24,10) senza mutare oggetti SWIG
- CustomRewardWrapper: fonde Monitor in __slots__, seed deterministici e cattura bytearray
  delle azioni negli ultimi 20% degli step per l'Episode Cache in RAM
- Persistent Render Env Pool in RAM per evitare re-init SDL/Pygame tra run
"""

from __future__ import annotations

import inspect
import math
import os
import sys
import warnings
from typing import Any, Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

try:
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize
    _GYM_AVAILABLE = True
except ImportError:
    gym = Any  # type: ignore
    DummyVecEnv = Any  # type: ignore
    VecNormalize = Any  # type: ignore
    VecEnv = Any  # type: ignore
    _GYM_AVAILABLE = False

from play2learn_lunarlander.config import (
    BOUNDARY_X,
    FUEL_MAIN_ENGINE,
    FUEL_SIDE_ENGINE,
    HELIPAD_X_MAX,
    SAFE_LANDING_ANGLE,
    SAFE_MAX_VX,
    SAFE_MAX_VY,
    VECNORM_PARAMS,
)
from play2learn_core.agent import ShmChunkedVecEnv, ChunkedSubprocVecEnv

_ISFINITE = math.isfinite

# Parametri solver Box2D condivisi tra training e replay per determinismo bit-exact
FAST_BOX2D_VEL_ITERS: int = 24
FAST_BOX2D_POS_ITERS: int = 10


class _NullParticle:
    """Singleton no-op che evita creazione/simulazione di b2Body per il fumo in headless."""

    __slots__ = ()

    def ApplyLinearImpulse(self, *args, **kwargs) -> None:
        pass


_NULL_PARTICLE = _NullParticle()


class _FastWorldProxy:
    """
    Proxy leggero in __slots__ attorno a Box2D.b2World.
    Intercetta Step(dt, 180, 60) -> Step(dt, 24, 10) evitando AttributeError su SWIG.
    """

    __slots__ = ("_w", "_step", "_vi", "_pi")

    def __init__(self, world: Any, vel_iters: int = FAST_BOX2D_VEL_ITERS, pos_iters: int = FAST_BOX2D_POS_ITERS):
        object.__setattr__(self, "_w", world)
        object.__setattr__(self, "_step", world.Step)
        object.__setattr__(self, "_vi", vel_iters)
        object.__setattr__(self, "_pi", pos_iters)

    def Step(self, dt: float, vel_iters: int = FAST_BOX2D_VEL_ITERS, pos_iters: int = FAST_BOX2D_POS_ITERS):
        return self._step(dt, self._vi, self._pi)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._w, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._w, name, value)


def patch_lunarlander_physics(
    env: gym.Env,
    disable_particles: bool = True,
    fast_solver: bool = True,
    vel_iters: int = FAST_BOX2D_VEL_ITERS,
    pos_iters: int = FAST_BOX2D_POS_ITERS,
) -> gym.Env:
    """Applica le ottimizzazioni Box2D all'istanza LunarLander unwrapped."""
    unwrapped = env.unwrapped
    if disable_particles and getattr(unwrapped, "render_mode", None) is None:
        if hasattr(unwrapped, "_create_particle"):
            unwrapped._create_particle = lambda *args, **kwargs: _NULL_PARTICLE

    if fast_solver and not getattr(unwrapped, "_p2l_fast_solver", False):
        orig_reset = unwrapped.reset

        def _fast_reset(*args, **kwargs):
            res = orig_reset(*args, **kwargs)
            w = getattr(unwrapped, "world", None)
            if w is not None and not isinstance(w, _FastWorldProxy):
                unwrapped.world = _FastWorldProxy(w, vel_iters=vel_iters, pos_iters=pos_iters)
            return res

        unwrapped.reset = _fast_reset
        unwrapped._p2l_fast_solver = True
    return env


def _strip_order_enforcing(env: gym.Env) -> gym.Env:
    """Rimuove il wrapper OrderEnforcing per risparmiare 1 chiamata Python/step."""
    if env.__class__.__name__ == "OrderEnforcing" and hasattr(env, "env"):
        return env.env
    return env


# Pool RAM persistente per ambienti render_mode="rgb_array" (evita re-init Pygame/SDL)
_RENDER_ENV_POOL: dict[tuple, gym.Env] = {}


def get_cached_render_env(
    env_id: str,
    env_kwargs: dict[str, Any] | None = None,
    fast_solver: bool = True,
) -> gym.Env:
    """Restituisce un ambiente rgb_array pre-riscaldato dalla cache RAM."""
    env_kwargs = env_kwargs or {}
    key = (env_id, tuple(sorted(env_kwargs.items())), fast_solver)
    env = _RENDER_ENV_POOL.get(key)
    if env is not None:
        return env
    try:
        env = gym.make(env_id, render_mode="rgb_array", disable_env_checker=True, **env_kwargs)
    except TypeError:
        env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
    env = _strip_order_enforcing(env)
    # In rendering manteniamo le particelle visibili ma allineiamo il solver per replay esatto
    patch_lunarlander_physics(env, disable_particles=False, fast_solver=fast_solver)
    _RENDER_ENV_POOL[key] = env
    return env


def classify_lunar_lander_outcome(
    obs: Any,
    original_reward: float,
    terminated: bool,
    truncated: bool,
    unwrapped: Any = None,
) -> tuple[bool, bool, bool, bool]:
    """Restituisce: (is_success, is_crashed, is_timeout, is_off_target)."""
    is_timeout = bool(truncated and not terminated)
    if not terminated:
        return False, False, is_timeout, False

    game_over = bool(getattr(unwrapped, "game_over", False))
    lander = getattr(unwrapped, "lander", None)
    is_awake = bool(getattr(lander, "awake", True)) if lander is not None else True

    final_x = abs(float(obs.item(0)))
    final_y = abs(float(obs.item(1)))
    final_vx = abs(float(obs.item(2)))
    final_vy = abs(float(obs.item(3)))
    final_angle = abs(float(obs.item(4)))
    both_legs = (obs.item(6) > 0.5) and (obs.item(7) > 0.5)

    if game_over or final_x >= BOUNDARY_X:
        return False, True, False, (final_x > HELIPAD_X_MAX)

    on_helipad = (final_x <= HELIPAD_X_MAX) and (final_y <= 0.15)
    upright = final_angle <= SAFE_LANDING_ANGLE
    settled = (not is_awake) or (final_vx <= SAFE_MAX_VX and final_vy <= SAFE_MAX_VY)
    gym_ok = (not is_awake) or (original_reward >= 50.0)

    if on_helipad and upright and both_legs and settled and gym_ok:
        return True, False, False, False

    return False, True, False, (final_x > HELIPAD_X_MAX)


def _build_reward_caller(reward_fn: Callable) -> Callable:
    try:
        sig = inspect.signature(reward_fn)
        params = list(sig.parameters.values())
        has_var = any(
            p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for p in params
        )
        param_count = -1 if has_var else len(params)
    except Exception:
        param_count = 5

    if param_count == 5:
        return reward_fn
    if param_count == 4:
        return lambda obs, act, succ, crash, tout: reward_fn(obs, succ, crash, tout)
    if param_count == 2:
        return lambda obs, act, succ, crash, tout: reward_fn(obs, act)
    if param_count == 1:
        return lambda obs, act, succ, crash, tout: reward_fn(obs)
    if param_count == 12:
        return lambda obs, act, succ, crash, tout: reward_fn(*obs, act, succ, crash, tout)
    if param_count == 11:
        return lambda obs, act, succ, crash, tout: reward_fn(*obs, succ, crash, tout)

    def _fallback(obs, act, succ, crash, tout):
        try:
            return reward_fn(obs, act, succ, crash, tout)
        except TypeError:
            return reward_fn(obs, succ, crash, tout)

    return _fallback


class CustomRewardWrapper(gym.Wrapper if _GYM_AVAILABLE else object):
    """
    Wrapper fuso (sostituisce anche SB3 Monitor) con:
    - Tracking deterministico del seed di ogni episodio per replay esatto da RAM
    - Cattura bytearray ultra-leggera delle azioni nell'ultimo 20% di training
    - Zero allocazioni dict su step non terminali
    """

    __slots__ = (
        "reward_fn",
        "_call_reward",
        "_unwrapped",
        "_seed_base",
        "_ep_counter",
        "_current_seed",
        "_cache_start_step",
        "_total_steps",
        "_recording_ep",
        "_ep_actions",
        "ep_reward",
        "ep_steps",
        "ep_fuel_used",
        "ep_main_engine_steps",
        "ep_side_engine_steps",
        "ep_air_steps",
        "ep_ground_steps",
    )

    def __init__(
        self,
        env: gym.Env,
        reward_fn: Callable,
        seed_base: int = 42,
        cache_start_step: int = 0,
    ):
        super().__init__(env)
        self.reward_fn = reward_fn
        self._call_reward = _build_reward_caller(reward_fn)
        self._unwrapped = env.unwrapped
        self._seed_base = int(seed_base)
        self._ep_counter = 0
        self._current_seed = int(seed_base)
        self._cache_start_step = int(cache_start_step)
        self._total_steps = 0
        self._recording_ep = False
        self._ep_actions = bytearray()

        self.ep_reward = 0.0
        self.ep_steps = 0
        self.ep_fuel_used = 0.0
        self.ep_main_engine_steps = 0
        self.ep_side_engine_steps = 0
        self.ep_air_steps = 0
        self.ep_ground_steps = 0

    def reset(self, **kwargs):
        self.ep_reward = 0.0
        self.ep_steps = 0
        self.ep_fuel_used = 0.0
        self.ep_main_engine_steps = 0
        self.ep_side_engine_steps = 0
        self.ep_air_steps = 0
        self.ep_ground_steps = 0

        seed = kwargs.get("seed")
        if seed is None:
            seed = self._seed_base + self._ep_counter
            kwargs["seed"] = seed
        self._current_seed = int(seed)
        self._ep_counter += 1

        self._recording_ep = (
            self._cache_start_step >= 0 and self._total_steps >= self._cache_start_step
        )
        if self._recording_ep:
            self._ep_actions.clear()

        return self.env.reset(**kwargs)

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        act_int = action if type(action) is int else int(action)

        self._total_steps += 1
        self.ep_steps += 1
        if self._recording_ep:
            self._ep_actions.append(act_int)

        if act_int == 2:
            self.ep_fuel_used += FUEL_MAIN_ENGINE
            self.ep_main_engine_steps += 1
        elif act_int == 1 or act_int == 3:
            self.ep_fuel_used += FUEL_SIDE_ENGINE
            self.ep_side_engine_steps += 1

        if obs.item(6) > 0.5 or obs.item(7) > 0.5:
            self.ep_ground_steps += 1
        else:
            self.ep_air_steps += 1

        if not (terminated or truncated):
            try:
                custom_rew = float(self._call_reward(obs, act_int, False, False, False))
                if not _ISFINITE(custom_rew):
                    raise ValueError(f"Valore non finito restituito: {custom_rew} (NaN o Inf).")
            except Exception as e:
                raise RuntimeError(
                    f"[Play2Learn Error] Errore nell'esecuzione della tua 'custom_reward': {e}\n"
                    f"Controlla i calcoli matematici (es. divisioni per zero, log di negativi o indici errati)."
                ) from e
            self.ep_reward += custom_rew
            return obs, custom_rew, False, False, info

        # Stato terminale / timeout (eseguito 1 volta per episodio)
        is_success, is_crashed, is_timeout, is_off_target = classify_lunar_lander_outcome(
            obs=obs,
            original_reward=float(original_reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            unwrapped=self._unwrapped,
        )

        try:
            custom_rew = float(self._call_reward(obs, act_int, is_success, is_crashed, is_timeout))
            if not _ISFINITE(custom_rew):
                raise ValueError(f"Valore non finito restituito: {custom_rew} (NaN o Inf).")
        except Exception as e:
            raise RuntimeError(
                f"[Play2Learn Error] Errore nell'esecuzione della tua 'custom_reward': {e}\n"
                f"Controlla i calcoli matematici (es. divisioni per zero, log di negativi o indici errati)."
            ) from e

        self.ep_reward += custom_rew

        # Emula SB3 Monitor senza il wrapper aggiuntivo
        info["episode"] = {
            "r": self.ep_reward,
            "l": self.ep_steps,
            "is_success": is_success,
        }
        info["is_success"] = is_success
        info["is_crashed"] = is_crashed
        info["is_timeout"] = is_timeout
        info["is_off_target"] = is_off_target

        info["telemetry"] = {
            "fuel_used": self.ep_fuel_used,
            "main_engine_steps": self.ep_main_engine_steps,
            "side_engine_steps": self.ep_side_engine_steps,
            "air_steps": self.ep_air_steps,
            "ground_steps": self.ep_ground_steps,
            "final_abs_x": abs(float(obs.item(0))),
            "final_abs_y": abs(float(obs.item(1))),
            "final_abs_vx": abs(float(obs.item(2))),
            "final_abs_vy": abs(float(obs.item(3))),
            "final_abs_angle": abs(float(obs.item(4))),
        }

        if self._recording_ep and self._ep_actions:
            info["replay"] = (self._current_seed, bytes(self._ep_actions))

        return obs, custom_rew, terminated, truncated, info


def _make_single_env(
    env_id: str,
    reward_fn: Callable,
    seed: int,
    rank: int,
    env_kwargs: dict[str, Any],
    cache_start_step: int,
    fast_solver: bool = True,
):
    def _init() -> gym.Env:
        try:
            env = gym.make(env_id, disable_env_checker=True, **env_kwargs)
        except TypeError:
            env = gym.make(env_id, **env_kwargs)
        env = _strip_order_enforcing(env)
        patch_lunarlander_physics(env, disable_particles=True, fast_solver=fast_solver)
        seed_base = seed + rank * 100_003
        env = CustomRewardWrapper(
            env,
            reward_fn,
            seed_base=seed_base,
            cache_start_step=cache_start_step,
        )
        env.reset(seed=seed_base)
        return env

    return _init


def get_optimal_vec_config(
    n_envs: int | None = None,
    total_timesteps: int = 200_000,
    prefer_subproc: bool | None = None,
) -> tuple[int, bool]:
    cpu_count = os.cpu_count() or 2
    is_linux = sys.platform.startswith("linux")
    use_subproc = prefer_subproc if prefer_subproc is not None else (is_linux and cpu_count >= 2)
    resolved_envs = 8 if n_envs is None else max(1, int(n_envs))
    if resolved_envs == 1:
        use_subproc = False
    return resolved_envs, use_subproc


def make_parallel_envs(
    env_id: str,
    reward_fn: Callable,
    n_envs: int | None = None,
    seed: int = 42,
    env_kwargs: dict[str, Any] | None = None,
    prefer_subproc: bool | None = None,
    total_timesteps: int = 200_000,
    cache_tail_fraction: float = 0.20,
    fast_solver: bool = True,
) -> VecEnv:
    env_kwargs = env_kwargs or {}
    resolved_envs, use_subproc = get_optimal_vec_config(
        n_envs=n_envs,
        total_timesteps=total_timesteps,
        prefer_subproc=prefer_subproc,
    )
    # Inizia a registrare (seed, actions) in RAM nell'ultimo 20% del training
    cache_start_step = max(0, int((total_timesteps * (1.0 - cache_tail_fraction)) / resolved_envs))
    env_fns = [
        _make_single_env(env_id, reward_fn, seed, i, env_kwargs, cache_start_step, fast_solver=fast_solver)
        for i in range(resolved_envs)
    ]

    if use_subproc:
        try:
            return ShmChunkedVecEnv(env_fns, n_workers=min(2, resolved_envs))
        except Exception:
            return DummyVecEnv(env_fns)

    return DummyVecEnv(env_fns)


def wrap_vec_normalize(vec_env: VecEnv, gamma: float | None = None) -> VecEnv:
    params = dict(VECNORM_PARAMS)
    if gamma is not None:
        params["gamma"] = float(gamma)
    return VecNormalize(vec_env, **params)
