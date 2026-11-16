"""
Modulo di gestione dell'ambiente di simulazione Gymnasium e del CustomRewardWrapper.
Ottimizzato per massimo throughput su Google Colab (2 vCPU) e PC locali Windows:
- Classificatore unificato e rigoroso degli esiti di allunaggio (classify_lunar_lander_outcome)
- Dispatch pre-compilato della reward_fn (zero if/elif per step)
- Protezione da valori non finiti (NaN o Inf) per preservare VecNormalize
- Zero allocazioni/scritture su `info` nei passi non terminali
- Rimozione di PassiveEnvChecker durante il training
- Auto-tuning intelligente di DummyVecEnv vs SubprocVecEnv
- Supporto a VecNormalize(norm_obs=False, norm_reward=True)
"""

from __future__ import annotations

import os
import sys
import inspect
import math
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
    SAFE_MAX_VX,
    SAFE_MAX_VY,
    VECNORM_PARAMS,
)


def classify_lunar_lander_outcome(
    obs: Any,
    original_reward: float,
    terminated: bool,
    truncated: bool,
    unwrapped: Any = None,
) -> tuple[bool, bool, bool, bool]:
    """
    Unica fonte di verità per la classificazione dell'esito dell'episodio di volo.
    Restituisce: (is_success, is_crashed, is_timeout, is_off_target).
    """
    is_timeout = bool(truncated and not terminated)
    if not terminated:
        return False, False, is_timeout, False

    game_over = bool(getattr(unwrapped, "game_over", False))
    lander = getattr(unwrapped, "lander", None)
    is_awake = bool(getattr(lander, "awake", True)) if lander is not None else True

    try:
        final_x = abs(float(obs.item(0)))
        final_y = abs(float(obs.item(1)))
        final_vx = abs(float(obs.item(2)))
        final_vy = abs(float(obs.item(3)))
        final_angle = abs(float(obs.item(4)))
        both_legs = (float(obs.item(6)) > 0.5) and (float(obs.item(7)) > 0.5)
    except (AttributeError, ValueError):
        final_x = abs(float(obs[0]))
        final_y = abs(float(obs[1]))
        final_vx = abs(float(obs[2]))
        final_vy = abs(float(obs[3]))
        final_angle = abs(float(obs[4]))
        both_legs = (float(obs[6]) > 0.5) and (float(obs[7]) > 0.5)

    # 1. Schianto immediato scafo o fuori schermo
    if game_over or final_x >= BOUNDARY_X:
        return False, True, False, (final_x > HELIPAD_X_MAX)

    # 2. Criteri fisici e didattici rigorosi per un vero allunaggio morbido:
    #    - Entro le bandierine (|x| <= 0.20) e a quota piazzola (|y| <= 0.15)
    #    - Assetto verticale (|angle| <= 0.25 rad, ~14°)
    #    - Entrambe le zampe saldamente a contatto
    #    - Lander fermo (Box2D sleep) o velocità residue entro i margini di sicurezza
    #    - Segnale Gym non distruttivo
    on_helipad = (final_x <= HELIPAD_X_MAX) and (final_y <= 0.15)
    upright = final_angle <= SAFE_LANDING_ANGLE
    settled = (not is_awake) or (final_vx <= SAFE_MAX_VX and final_vy <= SAFE_MAX_VY)
    gym_ok = (not is_awake) or (original_reward >= 50.0)

    if on_helipad and upright and both_legs and settled and gym_ok:
        return True, False, False, False

    return False, True, False, (final_x > HELIPAD_X_MAX)


def _build_reward_caller(reward_fn: Callable) -> Callable:
    """
    Ispeziona una sola volta la firma della funzione dello studente e restituisce
    un dispatcher specializzato ad alta velocità (zero controlli in step()).
    """
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
    Wrapper ad alte prestazioni che intercetta ogni `step()`, calcola la telemetria
    di volo, valida i valori numerici e delega il calcolo della ricompensa.
    Popola il dizionario `info` solo a fine episodio per minimizzare allocazioni e IPC.
    """

    __slots__ = (
        "reward_fn",
        "_call_reward",
        "_unwrapped",
        "ep_fuel_used",
        "ep_main_engine_steps",
        "ep_side_engine_steps",
        "ep_air_steps",
        "ep_ground_steps",
    )

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

        contact = (obs[6] > 0.5) or (obs[7] > 0.5)

        if contact:
            self.ep_ground_steps += 1
        else:
            self.ep_air_steps += 1

        # 2. Fast-path per oltre il 99.5% degli step (non terminali)
        if not (terminated or truncated):
            try:
                custom_rew = float(self._call_reward(obs, act_int, False, False, False))
                if not math.isfinite(custom_rew):
                    raise ValueError(f"Valore non finito restituito: {custom_rew} (NaN o Inf).")
            except Exception as e:
                raise RuntimeError(
                    f"[Play2Learn Error] Errore nell'esecuzione della tua 'custom_reward': {e}\n"
                    f"Controlla i calcoli matematici (es. divisioni per zero, log di negativi o indici errati)."
                ) from e
            return obs, custom_rew, False, False, info

        # 3. Stato terminale / timeout (eseguito 1 sola volta per episodio)
        is_success, is_crashed, is_timeout, is_off_target = classify_lunar_lander_outcome(
            obs=obs,
            original_reward=float(original_reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            unwrapped=self._unwrapped,
        )

        try:
            custom_rew = float(self._call_reward(obs, act_int, is_success, is_crashed, is_timeout))
            if not math.isfinite(custom_rew):
                raise ValueError(f"Valore non finito restituito: {custom_rew} (NaN o Inf).")
        except Exception as e:
            raise RuntimeError(
                f"[Play2Learn Error] Errore nell'esecuzione della tua 'custom_reward': {e}\n"
                f"Controlla i calcoli matematici (es. divisioni per zero, log di negativi o indici errati)."
            ) from e

        # Popolamento `info` concentrato a fine episodio
        info["custom_reward"] = custom_rew
        info["original_gym_reward"] = float(original_reward)
        info["is_success"] = is_success
        info["is_crashed"] = is_crashed
        info["is_timeout"] = is_timeout
        info["is_off_target"] = is_off_target

        try:
            fx = abs(float(obs.item(0)))
            fy = abs(float(obs.item(1)))
            fvx = abs(float(obs.item(2)))
            fvy = abs(float(obs.item(3)))
            fa = abs(float(obs.item(4)))
        except (AttributeError, ValueError):
            fx = abs(float(obs[0]))
            fy = abs(float(obs[1]))
            fvx = abs(float(obs[2]))
            fvy = abs(float(obs[3]))
            fa = abs(float(obs[4]))

        info["telemetry"] = {
            "fuel_used": float(self.ep_fuel_used),
            "main_engine_steps": self.ep_main_engine_steps,
            "side_engine_steps": self.ep_side_engine_steps,
            "air_steps": self.ep_air_steps,
            "ground_steps": self.ep_ground_steps,
            "final_abs_x": fx,
            "final_abs_y": fy,
            "final_abs_vx": fvx,
            "final_abs_vy": fvy,
            "final_abs_angle": fa,
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
    - Colab / CPU <= 4 thread: DummyVecEnv con 8 env (massimo throughput, zero IPC).
    - Windows (spawn lento e fragile in Jupyter): DummyVecEnv con 8 env di default.
    - Linux multi-core (>= 6 thread) su run >= 250k step: SubprocVecEnv con 8 env.
    """
    cpu_count = os.cpu_count() or 2
    is_windows = sys.platform.startswith("win")

    if prefer_subproc is not None:
        use_subproc = prefer_subproc
    else:
        use_subproc = (not is_windows) and (cpu_count >= 6) and (total_timesteps >= 250_000)

    if n_envs is None:
        resolved_envs = 8
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
