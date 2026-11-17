"""
Base Wrapper per Ambienti Didattici di Reinforcement Learning.
Fornisce ispezione rapida della firma dello studente, validazione matematica (NaN/Inf)
e tracciamento unificato della telemetria.
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Callable
import gymnasium as gym


def build_reward_caller(reward_fn: Callable) -> Callable:
    """
    Ispeziona una sola volta all'inizializzazione la firma della funzione dello studente
    e restituisce un dispatcher specializzato ad altissima velocità.
    Supporta:
      - 5 argomenti: (obs, action, is_success, is_crashed, is_timeout)
      - 4 argomenti: (obs, is_success, is_crashed, is_timeout)
      - 2 argomenti: (obs, action)
      - 1 argomento:  (obs)
      - Generico / **kwargs
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

    def _safe_call(obs: Any, act: Any, succ: bool, crash: bool, tout: bool) -> float:
        kwargs: dict[str, Any] = {
            "obs": obs,
            "action": act,
            "is_success": succ,
            "is_crashed": crash,
            "is_timeout": tout,
        }
        try:
            return float(reward_fn(**kwargs))
        except TypeError:
            try:
                return float(reward_fn(obs, act))
            except TypeError:
                return float(reward_fn(obs))

    return _safe_call


class BaseRewardWrapper(gym.Wrapper):
    """
    Classe base per tutti i wrapper didattici Play2Learn.
    Isola l'algoritmo SB3 (PPO) dagli errori dello studente e valida
    la correttezza numerica di ogni passo di simulazione.
    """

    def __init__(self, env: gym.Env, reward_fn: Callable | None = None):
        super().__init__(env)
        self.reward_fn = reward_fn
        self._caller = build_reward_caller(reward_fn) if reward_fn is not None else None
        self.current_step = 0

    def validate_reward_value(self, reward_val: Any) -> float:
        """Converte e valida la reward dello studente con messaggi didattici chiari."""
        if reward_val is None:
            raise ValueError(
                "La tua funzione 'custom_reward' ha restituito None. "
                "Assicurati di terminare con 'return float(reward)'."
            )

        # Supporto sicuro per tensori PyTorch e array NumPy scalari
        if hasattr(reward_val, "detach") and hasattr(reward_val, "numel"):
            if reward_val.numel() != 1:
                raise ValueError(
                    f"Hai restituito un tensore PyTorch di forma {tuple(reward_val.shape)}: "
                    "la reward deve essere un singolo numero scalare."
                )
            reward_val = reward_val.detach().cpu().item()
        elif hasattr(reward_val, "shape") and hasattr(reward_val, "item"):
            if getattr(reward_val, "size", 0) != 1:
                raise ValueError(
                    f"Hai restituito un array NumPy di forma {reward_val.shape}: "
                    "la reward deve essere un singolo numero scalare."
                )
            reward_val = reward_val.item()

        try:
            val = float(reward_val)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"La funzione di reward ha restituito un tipo non numerico ({type(reward_val).__name__}): {reward_val}"
            ) from exc

        if not math.isfinite(val):
            raise ValueError(
                f"Valore non finito restituito: {val} (NaN o Inf). "
                "Verifica eventuali divisioni per zero o logaritmi di numeri negativi."
            )
        return val

    def call_student_reward(
        self,
        obs: Any,
        action: Any,
        is_success: bool,
        is_crashed: bool,
        is_timeout: bool
    ) -> float:
        """Invoca la funzione dello studente proteggendo da crash opachi."""
        if self._caller is None:
            return 0.0

        try:
            raw_rew = self._caller(obs, action, is_success, is_crashed, is_timeout)
            return self.validate_reward_value(raw_rew)
        except Exception as exc:
            raise RuntimeError(
                f"[Play2Learn Error] Errore nell'esecuzione della tua 'custom_reward': {exc}\n"
                "Controlla i calcoli matematici (es. divisioni per zero, log di negativi o indici errati)."
            ) from exc
