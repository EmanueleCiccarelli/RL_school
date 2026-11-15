"""
Modulo di gestione dell'ambiente di simulazione Gymnasium e del CustomRewardWrapper.
Ottimizzato per massimo throughput su Google Colab (2 vCPU) e PC locali Windows:
- Dispatch pre-compilato della reward_fn (zero if/elif per step)
- Zero allocazioni/scritture su `info` nei passi non terminali
- Rimozione di PassiveEnvChecker durante il training
- Auto-tuning intelligente di DummyVecEnv vs SubprocVecEnv
- Supporto a VecNormalize(norm_obs=False, norm_reward=True)
"""

from __future__ import annotations

import os
import sys
import inspect
import warnings
from typing import Callable, Any

# Evita contesa di thread BLAS/OpenMP tra PyTorch e Box2D su CPU vincolate
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

try:
    import gymnasium as gym
    from stable_baselines3.common.vec_env import (
        SubprocVecEnv,
        DummyVecEnv,
        VecNormalize,
        VecEnv,
    )
    from stable_baselines3.common.monitor import Monitor
    _GYM_AVAILABLE = True
except ImportError:
    gym = Any  # type: ignore
    SubprocVecEnv = Any  # type: ignore
    DummyVecEnv = Any  # type: ignore
    VecNormalize = Any  # type: ignore
    VecEnv = Any  # type: ignore
    Monitor = Any  # type: ignore
    _GYM_AVAILABLE = False

from play2learn_lunarlander.config import (
    FUEL_MAIN_ENGINE,
    FUEL_SIDE_ENGINE,
    BOUNDARY_X,
    HELIPAD_X_MAX,
    SAFE_LANDING_ANGLE,
    VECNORM_PARAMS,
)


def _build_reward_caller(reward_fn: Callable) -> Callable:
    """
    Ispeziona una sola volta la firma della funzione dello studente e restituisce
    un dispatcher specializzato ad alta velocità (zero controlli in step()).
    """
    try:
        param_count = len(inspect.signature(reward_fn).parameters)
    except Exception:
        param_count = 5

    if param_count == 5:
        return lambda obs, act, succ, crash, tout: reward_fn(obs, act, succ, crash, tout)
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
    Wrapper ad alte prestazioni che intercetta ogni `step()`, calcola la telemetria
    di volo e delega il calcolo della ricompensa alla funzione dello studente.
    Popola il dizionario `info` solo a fine episodio per minimizzare allocazioni e IPC.
    """

    def __init__(self, env: gym.Env, reward_fn: Callable):
        super().__init__(env)
        self.reward_fn = reward_fn
        self._call_reward = _build_reward_caller(reward_fn)
        self._unwrapped = env.unwrapped

        self.ep_fuel_used: float = 0.0
        self.ep_main_engine_steps: int = 0
        self.ep_side_engine_steps: int = 0
        self.ep_air_steps: int = 0
        self.ep_ground_steps: int = 0

    def reset(self, **kwargs):
        self.ep_fuel_used = 0.0
        self.ep_main_engine_steps = 0
        self.ep_side_engine_steps = 0
        self.ep_air_steps = 0
        self.ep_ground_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        act_int = int(action)

        # 1. Telemetria leggera in-place (zero allocazioni)
        if act_int == 2:
            self.ep_fuel_used += FUEL_MAIN_ENGINE
            self.ep_main_engine_steps += 1
        elif act_int == 1 or act_int == 3:
            self.ep_fuel_used += FUEL_SIDE_ENGINE
            self.ep_side_engine_steps += 1

        # Accesso scalare diretto (.item) evita allocazione di np.float32
        if obs.item(6) > 0.5 or obs.item(7) > 0.5:
            self.ep_ground_steps += 1
        else:
            self.ep_air_steps += 1

        # 2. Fast-path per oltre il 99.5% degli step (non terminali)
        if not (terminated or truncated):
            try:
                custom_rew = float(self._call_reward(obs, act_int, False, False, False))
            except Exception as e:
                raise RuntimeError(
                    f"[Play2Learn Error] Errore nell'esecuzione della tua 'custom_reward': {e}\n"
                    f"Controlla i calcoli matematici (es. divisioni per zero o indici errati)."
                ) from e
            return obs, custom_rew, False, False, info

        # 3. Stato terminale / timeout (eseguito 1 volta per episodio)
        is_timeout = bool(truncated)
        is_crashed = False
        is_success = False
        is_off_target = False

        if terminated:
            unwrapped = self._unwrapped
            game_over = getattr(unwrapped, "game_over", False)
            lander = getattr(unwrapped, "lander", None)
            is_awake = getattr(lander, "awake", True) if lander is not None else True

            final_x = abs(obs.item(0))
            final_angle = abs(obs.item(4))
            legs_contact = (obs.item(6) > 0.5) or (obs.item(7) > 0.5)

            # 1. Schianto catastrofico evidente: scafo tocca terra (game_over) o fuori schermo (|x| >= 1.0)
            if game_over or final_x >= BOUNDARY_X:
                is_crashed = True
                is_success = False
            # 2. Il modulo ha toccato terra e si è fermato (Box2D sleep / awake=False) o fine episodio positiva Gym
            elif (not is_awake) or (original_reward >= 100.0):
                # Per essere un vero SUCCESSO didattico e fisico, l'atterraggio DEVE avvenire:
                # - Sulla piazzola delimitata dalle bandierine (|x| <= 0.20)
                # - Con assetto verticale stabile (|angle| <= 0.25 rad, ~14°)
                # - Con contatto delle zampe
                on_helipad = final_x <= HELIPAD_X_MAX
                upright = final_angle <= SAFE_LANDING_ANGLE

                if on_helipad and upright and legs_contact:
                    is_success = True
                    is_crashed = False
                else:
                    # Atterraggio completato ma fuori piazzola (es. su crateri/pendii) o ribaltato:
                    # è un fallimento/fuori bersaglio!
                    is_success = False
                    is_crashed = True
                    if not on_helipad:
                        is_off_target = True
            else:
                # Caso residuale
                on_helipad = final_x <= HELIPAD_X_MAX
                upright = final_angle <= SAFE_LANDING_ANGLE
                if on_helipad and upright and legs_contact and original_reward > 0.0:
                    is_success = True
                    is_crashed = False
                else:
                    is_crashed = True
                    is_success = False
                    if not on_helipad:
                        is_off_target = True

        try:
            custom_rew = float(
                self._call_reward(obs, act_int, is_success, is_crashed, is_timeout)
            )
        except Exception as e:
            raise RuntimeError(
                f"[Play2Learn Error] Errore nell'esecuzione della tua 'custom_reward': {e}\n"
                f"Controlla i calcoli matematici (es. divisioni per zero o indici errati)."
            ) from e

        # Popolamento `info` concentrato a fine episodio
        info["custom_reward"] = custom_rew
        info["original_gym_reward"] = float(original_reward)
        info["is_success"] = is_success
        info["is_crashed"] = is_crashed
        info["is_timeout"] = is_timeout
        info["is_off_target"] = is_off_target
        info["telemetry"] = {
            "fuel_used": float(self.ep_fuel_used),
            "main_engine_steps": self.ep_main_engine_steps,
            "side_engine_steps": self.ep_side_engine_steps,
            "air_steps": self.ep_air_steps,
            "ground_steps": self.ep_ground_steps,
            "final_abs_x": abs(obs.item(0)),
            "final_abs_y": abs(obs.item(1)),
            "final_abs_vx": abs(obs.item(2)),
            "final_abs_vy": abs(obs.item(3)),
            "final_abs_angle": abs(obs.item(4)),
        }

        return obs, custom_rew, terminated, truncated, info


def _make_single_env(
    env_id: str,
    reward_fn: Callable,
    seed: int,
    rank: int,
    env_kwargs: dict[str, Any],
):
    """Costruisce un singolo ambiente isolato, senza PassiveEnvChecker."""
    def _init() -> gym.Env:
        try:
            env = gym.make(env_id, disable_env_checker=True, **env_kwargs)
        except TypeError:
            env = gym.make(env_id, **env_kwargs)
        env = CustomRewardWrapper(env, reward_fn)
        env = Monitor(env, info_keywords=("is_success",))
        env.reset(seed=seed + rank)
        return env
    return _init


def get_optimal_vec_config(
    n_envs: int | None = None,
    total_timesteps: int = 200_000,
    prefer_subproc: bool | None = None,
) -> tuple[int, bool]:
    """
    Determina la combinazione ottimale (n_envs, use_subproc) per l'hardware corrente:
    - Colab / CPU <= 4 thread: DummyVecEnv con 4 env (massimo FPS, zero IPC).
    - Windows (spawn lento e fragile in Jupyter): DummyVecEnv con 4 env di default.
    - Linux multi-core (>= 6 thread) su run >= 250k step: SubprocVecEnv con 8 env.
    """
    cpu_count = os.cpu_count() or 2
    is_windows = sys.platform.startswith("win")

    if prefer_subproc is not None:
        use_subproc = prefer_subproc
    else:
        use_subproc = (not is_windows) and (cpu_count >= 6) and (total_timesteps >= 250_000)

    if n_envs is None:
        resolved_envs = 8 if use_subproc else 4
    else:
        resolved_envs = max(1, int(n_envs))

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
) -> VecEnv:
    """
    Crea la VecEnv parallela ottimizzata per Google Colab (2 vCPU) e desktop locali.
    """
    env_kwargs = env_kwargs or {}
    resolved_envs, use_subproc = get_optimal_vec_config(
        n_envs=n_envs,
        total_timesteps=total_timesteps,
        prefer_subproc=prefer_subproc,
    )
    env_fns = [
        _make_single_env(env_id, reward_fn, seed, i, env_kwargs)
        for i in range(resolved_envs)
    ]

    if use_subproc:
        try:
            start_method = "forkserver" if not sys.platform.startswith("win") else "spawn"
            return SubprocVecEnv(env_fns, start_method=start_method)
        except Exception:
            return DummyVecEnv(env_fns)

    return DummyVecEnv(env_fns)


def wrap_vec_normalize(
    vec_env: VecEnv,
    gamma: float | None = None,
) -> VecEnv:
    """
    Applica VecNormalize SOLO sulle ricompense (norm_obs=False, norm_reward=True).
    Stabilizza PPO con qualsiasi scala di custom_reward senza alterare le osservazioni
    e mantenendo intatta la compatibilità a inferenza senza file di statistiche extra.
    """
    params = dict(VECNORM_PARAMS)
    if gamma is not None:
        params["gamma"] = float(gamma)
    return VecNormalize(vec_env, **params)
