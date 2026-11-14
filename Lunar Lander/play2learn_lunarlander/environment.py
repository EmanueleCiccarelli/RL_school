"""
Modulo di gestione dell'ambiente di simulazione Gymnasium e del CustomRewardWrapper.
Ottimizzato per Google Colab (gestione multi-threading e DummyVecEnv vs SubprocVecEnv).
"""

from __future__ import annotations

import os
import inspect
import warnings
from typing import Callable, Any

# Ottimizzazione thread per evitare contesa su macchine standard Colab (2 vCPU)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Silenzia i warning di deprecazione di Gym/Swig per non confondere gli studenti
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

try:
    import gymnasium as gym
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnv
    from stable_baselines3.common.monitor import Monitor
    _GYM_AVAILABLE = True
except ImportError:
    gym = Any  # type: ignore
    SubprocVecEnv = Any  # type: ignore
    DummyVecEnv = Any  # type: ignore
    VecEnv = Any  # type: ignore
    Monitor = Any  # type: ignore
    _GYM_AVAILABLE = False

from play2learn_lunarlander.config import (
    FUEL_MAIN_ENGINE,
    FUEL_SIDE_ENGINE,
    BOUNDARY_X,
)


class CustomRewardWrapper(gym.Wrapper if _GYM_AVAILABLE else object):
    """
    Wrapper che intercetta ogni `step()` dell'ambiente Gymnasium,
    calcola la telemetria di volo in tempo reale (consumo carburante, tempo in aria,
    motori azionati, velocità di impatto) e delega il calcolo della ricompensa alla
    funzione matematica scritta dallo studente.
    """

    def __init__(self, env: gym.Env, reward_fn: Callable):
        super().__init__(env)
        self.reward_fn = reward_fn

        # Ispezione della firma della funzione dello studente per massima flessibilità
        try:
            sig = inspect.signature(reward_fn)
            self._param_count = len(sig.parameters)
        except Exception:
            self._param_count = 5

        self._reset_telemetry()

    def _reset_telemetry(self) -> None:
        """Azzera i contatori telemetrici all'inizio di ogni episodio."""
        self.ep_fuel_used: float = 0.0
        self.ep_main_engine_steps: int = 0
        self.ep_side_engine_steps: int = 0
        self.ep_air_steps: int = 0
        self.ep_ground_steps: int = 0

    def reset(self, **kwargs):
        self._reset_telemetry()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        act_int = int(action)

        # 1. Telemetria del consumo di carburante e uso motori
        if act_int == 2:
            self.ep_fuel_used += FUEL_MAIN_ENGINE
            self.ep_main_engine_steps += 1
        elif act_int in (1, 3):
            self.ep_fuel_used += FUEL_SIDE_ENGINE
            self.ep_side_engine_steps += 1

        # Sensori zampe: obs[6] = zampa sinistra, obs[7] = zampa destra
        if obs[6] > 0.5 or obs[7] > 0.5:
            self.ep_ground_steps += 1
        else:
            self.ep_air_steps += 1

        # 2. Identificazione affidabile degli eventi terminali
        is_timeout = bool(truncated)
        is_crashed = False
        is_success = False

        if terminated:
            unwrapped = self.env.unwrapped
            game_over = getattr(unwrapped, "game_over", False)
            lander = getattr(unwrapped, "lander", None)

            # Caso schianto: game over scattato, uscita dallo schermo, o reward originaria di crash (<= -100)
            if game_over or abs(obs[0]) >= BOUNDARY_X or original_reward <= -100.0:
                is_crashed = True
            # Caso successo: lander a terra addormentato (dormiente) o reward originaria di successo (>= 100)
            elif (lander is not None and not lander.awake) or original_reward >= 100.0:
                is_success = True
            else:
                if original_reward > 0:
                    is_success = True
                else:
                    is_crashed = True

        # 3. Chiamata alla funzione dello studente con gestione eccezioni
        try:
            if self._param_count == 5:
                custom_rew = self.reward_fn(obs, act_int, is_success, is_crashed, is_timeout)
            elif self._param_count == 4:
                custom_rew = self.reward_fn(obs, is_success, is_crashed, is_timeout)
            elif self._param_count == 2:
                custom_rew = self.reward_fn(obs, act_int)
            elif self._param_count == 1:
                custom_rew = self.reward_fn(obs)
            elif self._param_count == 12:
                custom_rew = self.reward_fn(*obs, act_int, is_success, is_crashed, is_timeout)
            elif self._param_count == 11:
                custom_rew = self.reward_fn(*obs, is_success, is_crashed, is_timeout)
            else:
                try:
                    custom_rew = self.reward_fn(obs, act_int, is_success, is_crashed, is_timeout)
                except TypeError:
                    custom_rew = self.reward_fn(obs, is_success, is_crashed, is_timeout)
        except Exception as e:
            # Fornisce un messaggio didattico chiaro anziché un crash misterioso del worker
            raise RuntimeError(
                f"[Play2Learn Error] Errore nell'esecuzione della tua 'custom_reward': {e}\n"
                f"Controlla i calcoli matematici (es. divisioni per zero o indici errati)."
            ) from e

        # 4. Arricchimento del dizionario `info` per telemetria e diagnostica
        info["custom_reward"] = float(custom_rew)
        info["original_gym_reward"] = float(original_reward)
        info["is_success"] = is_success
        info["is_crashed"] = is_crashed
        info["is_timeout"] = is_timeout

        if terminated or truncated:
            info["telemetry"] = {
                "fuel_used": float(self.ep_fuel_used),
                "main_engine_steps": int(self.ep_main_engine_steps),
                "side_engine_steps": int(self.ep_side_engine_steps),
                "air_steps": int(self.ep_air_steps),
                "ground_steps": int(self.ep_ground_steps),
                "final_abs_x": float(abs(obs[0])),
                "final_abs_y": float(abs(obs[1])),
                "final_abs_vx": float(abs(obs[2])),
                "final_abs_vy": float(abs(obs[3])),
                "final_abs_angle": float(abs(obs[4])),
            }

        return obs, float(custom_rew), terminated, truncated, info


def _make_single_env(
    env_id: str,
    reward_fn: Callable,
    seed: int,
    rank: int,
    env_kwargs: dict[str, Any]
):
    """Costruisce un singolo ambiente isolato con wrapper e monitor."""
    def _init() -> gym.Env:
        env = gym.make(env_id, **env_kwargs)
        env = CustomRewardWrapper(env, reward_fn)
        env = Monitor(env, info_keywords=("is_success",))
        env.reset(seed=seed + rank)
        return env
    return _init


def make_parallel_envs(
    env_id: str,
    reward_fn: Callable,
    n_envs: int = 8,
    seed: int = 42,
    env_kwargs: dict[str, Any] | None = None,
    prefer_subproc: bool | None = None
) -> VecEnv:
    """
    Crea un gruppo di ambienti paralleli ottimizzati per l'hardware in uso.

    Nota Colab: Su istanze standard (2 vCPU), `DummyVecEnv` evita l'overhead
    di serializzazione e IPC tra processi separati, risultando sensibilmente
    più veloce di `SubprocVecEnv`. Se ci sono 4+ core disponibili o l'utente
    lo richiede esplicitamente, viene usato `SubprocVecEnv`.
    """
    env_kwargs = env_kwargs or {}
    env_fns = [_make_single_env(env_id, reward_fn, seed, i, env_kwargs) for i in range(n_envs)]

    cpu_count = os.cpu_count() or 2
    use_subproc = prefer_subproc if prefer_subproc is not None else (cpu_count >= 4)

    if n_envs > 1 and use_subproc:
        try:
            return SubprocVecEnv(env_fns)
        except Exception:
            return DummyVecEnv(env_fns)
    return DummyVecEnv(env_fns)
