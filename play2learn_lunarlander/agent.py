"""
Modulo per la gestione dell'agente RL (PPO), learning rate schedule,
telemetria O(1), best-model checkpointing in RAM e progress bar leggera per Colab/Jupyter.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecEnv
except ImportError:
    PPO = Any  # type: ignore
    BaseCallback = object  # type: ignore
    VecEnv = Any  # type: ignore

from play2learn_lunarlander.config import (
    DEFAULT_PPO_PARAMS,
    DEFAULT_LR_START,
    DEFAULT_LR_END,
    TARGET_ROLLOUT_SIZE,
)


class LinearSchedule:
    """
    Schedule lineare top-level pienamente serializzabile da pickle/cloudpickle
    per PPO.save() e PPO.load() senza warning.
    """

    def __init__(self, initial_value: float, final_value: float):
        self.initial_value = float(initial_value)
        self.final_value = float(final_value)

    def __call__(self, progress_remaining: float) -> float:
        return self.final_value + progress_remaining * (self.initial_value - self.final_value)


class RewardTrackerCallback(BaseCallback):
    """
    Callback ad alte prestazioni:
    - Traccia telemetria completa per Diagnostics e Visualizer.
    - Calcola la media mobile del successo in O(1) con deque.
    - Salva i pesi migliori SOLO in RAM (~60 KB) durante il training, azzerando
      l'I/O sincrono su disco; scrive best_model.zip una sola volta a fine run.
    - Cattura snapshot milestone uniformi in RAM per la generazione on-demand del timelapse.
    - Mostra una progress bar HTML/console throttled (ogni 1.5s) senza lag su Colab.
    """

    def __init__(
        self,
        best_model_save_dir: Path | str | None = None,
        total_timesteps: int = 0,
        show_progress: bool = True,
        progress_interval_sec: float = 1.5,
        window_size: int = 30,
        verbose: int = 0,
        capture_milestones: bool = True,
    ):
        super().__init__(verbose)
        self.best_model_save_dir = Path(best_model_save_dir) if best_model_save_dir else None
        self.total_timesteps = int(total_timesteps)
        self.show_progress = show_progress
        self.progress_interval_sec = float(progress_interval_sec)
        self.window_size = int(window_size)
        self.capture_milestones = capture_milestones

        # Liste telemetriche pubbliche (contratto con diagnostics.py e visualizer.py)
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

        # Finestra scorrevole O(1)
        self._window_outcomes: deque[int] = deque(maxlen=self.window_size)
        self._window_fuels: deque[float] = deque(maxlen=self.window_size)
        self._window_rewards: deque[float] = deque(maxlen=self.window_size)
        self._window_successes: int = 0
        self._window_fuel_sum: float = 0.0
        self._window_reward_sum: float = 0.0

        # Snapshot in RAM del miglior modello
        self.best_moving_success_rate: float = -1.0
        self._best_moving_fuel: float = float("inf")
        self._best_policy_state: dict[str, torch.Tensor] | None = None
        self.best_model_path: Path | None = None

        # Milestone snapshots per timelapse evoluzione (RAM pura)
        self.milestone_snapshots: list[tuple[int, dict[str, torch.Tensor]]] = []
        self._milestone_targets: list[int] = []
        if self.capture_milestones and self.total_timesteps > 0:
            fractions = [0.05, 0.25, 0.50, 0.75, 1.00]
            self._milestone_targets = [int(self.total_timesteps * f) for f in fractions]

        # Stato UI throttled
        self._t_start: float = 0.0
        self._last_ui_update: float = 0.0
        self._display_handle: Any = None
        self._use_html_progress: bool = False

    def _on_training_start(self) -> None:
        self._t_start = time.perf_counter()
        self._last_ui_update = self._t_start

        if self.show_progress and self.total_timesteps > 0:
            try:
                from IPython import get_ipython
                from IPython.display import HTML, display
                shell = get_ipython()
                if shell is not None and shell.__class__.__name__ in (
                    "ZMQInteractiveShell",
                    "Shell",  # Google Colab
                ):
                    self._use_html_progress = True
                    self._display_handle = display(
                        HTML(self._render_html_progress(0, 0.0, 0.0)),
                        display_id=True,
                    )
            except Exception:
                self._use_html_progress = False

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos:
            return True

        for info in infos:
            ep_info = info.get("episode")
            if ep_info is None:
                continue

            ep_r = float(ep_info["r"])
            ep_l = int(ep_info["l"])
            self.episode_rewards.append(ep_r)
            self.episode_lengths.append(ep_l)

            is_succ = bool(info.get("is_success", False))
            is_crash = bool(info.get("is_crashed", False))
            is_tout = bool(info.get("is_timeout", False))

            if is_succ:
                self.outcomes.append("success")
                self.successes += 1
                succ_flag = 1
            elif is_crash:
                self.outcomes.append("crash")
                self.crashes += 1
                succ_flag = 0
            elif is_tout:
                self.outcomes.append("timeout")
                self.timeouts += 1
                succ_flag = 0
            else:
                self.outcomes.append("unknown")
                succ_flag = 0

            tel = info.get("telemetry")
            if tel:
                fuel = float(tel.get("fuel_used", 0.0))
                self.fuel_used_list.append(fuel)
                self.air_steps_list.append(int(tel.get("air_steps", ep_l)))
                self.ground_steps_list.append(int(tel.get("ground_steps", 0)))
                self.final_x_list.append(float(tel.get("final_abs_x", 0.0)))
                self.final_vy_list.append(float(tel.get("final_abs_vy", 0.0)))
                self.final_angle_list.append(float(tel.get("final_abs_angle", 0.0)))
            else:
                fuel = 0.0
                self.fuel_used_list.append(0.0)
                self.air_steps_list.append(ep_l)
                self.ground_steps_list.append(0)
                self.final_x_list.append(0.0)
                self.final_vy_list.append(0.0)
                self.final_angle_list.append(0.0)

            # Finestra scorrevole O(1)
            if len(self._window_outcomes) == self.window_size:
                self._window_successes -= self._window_outcomes[0]
                self._window_fuel_sum -= self._window_fuels[0]
                self._window_reward_sum -= self._window_rewards[0]
            self._window_outcomes.append(succ_flag)
            self._window_fuels.append(fuel)
            self._window_rewards.append(ep_r)
            self._window_successes += succ_flag
            self._window_fuel_sum += fuel
            self._window_reward_sum += ep_r

            # Aggiornamento miglior modello in RAM (minimo 15 episodi per stabilità statistica)
            if len(self._window_outcomes) >= min(15, self.window_size):
                curr_rate = (self._window_successes / len(self._window_outcomes)) * 100.0
                curr_fuel = self._window_fuel_sum / len(self._window_fuels)
                is_better = (
                    curr_rate > self.best_moving_success_rate
                    or (curr_rate == self.best_moving_success_rate and curr_fuel < self._best_moving_fuel)
                )
                if is_better and curr_rate > 0.0:
                    self.best_moving_success_rate = curr_rate
                    self._best_moving_fuel = curr_fuel
                    model = getattr(self, "model", None)
                    if model is not None and hasattr(model, "policy"):
                        self._best_policy_state = {
                            k: v.detach().cpu().clone()
                            for k, v in model.policy.state_dict().items()
                        }

        # Snapshot milestone periodici (per timelapse evoluzione su richiesta)
        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "policy"):
            if self._milestone_targets and self.num_timesteps >= self._milestone_targets[0]:
                tgt = self._milestone_targets.pop(0)
                self.milestone_snapshots.append((
                    tgt,
                    {k: v.detach().cpu().clone() for k, v in model.policy.state_dict().items()}
                ))

        # UI throttled a intervallo fisso (default 1.5s) per zero overhead
        now = time.perf_counter()
        if self.show_progress and (now - self._last_ui_update) >= self.progress_interval_sec:
            self._last_ui_update = now
            self._update_progress_ui(now)

        return True

    def _on_training_end(self) -> None:
        # Se rimane l'ultimo snapshot del 100%, catturalo
        if self._milestone_targets:
            self.milestone_snapshots.append((
                self.num_timesteps,
                {k: v.detach().cpu().clone() for k, v in self.model.policy.state_dict().items()}
            ))
            self._milestone_targets.clear()

        if self.show_progress:
            self._update_progress_ui(time.perf_counter(), finished=True)

    @property
    def current_moving_success_rate(self) -> float:
        if not self._window_outcomes:
            return 0.0
        return (self._window_successes / len(self._window_outcomes)) * 100.0

    @property
    def current_moving_reward(self) -> float:
        if not self._window_rewards:
            return 0.0
        return self._window_reward_sum / len(self._window_rewards)

    @property
    def current_moving_fuel(self) -> float:
        if not self._window_fuels:
            return 0.0
        return self._window_fuel_sum / len(self._window_fuels)

    def flush_best_model_to_disk(
        self,
        keep_best_loaded_if_superior: bool = True,
        superiority_margin: float = 5.0,
    ) -> bool:
        """
        Scrive su disco `best_model.zip` una sola volta a fine training.
        Se il miglior modello supera la policy finale di almeno `superiority_margin` %,
        lo mantiene attivo in `self.model` per la successiva fase di valutazione.
        Restituisce True se il miglior modello è stato caricato.
        """
        if self._best_policy_state is None or self.best_model_save_dir is None:
            return False

        self.best_model_save_dir.mkdir(parents=True, exist_ok=True)
        save_p = self.best_model_save_dir / "best_model.zip"

        final_state = {
            k: v.detach().cpu().clone()
            for k, v in self.model.policy.state_dict().items()
        }

        self.model.policy.load_state_dict(self._best_policy_state)
        self.model.save(str(save_p))
        self.best_model_path = save_p

        final_rate = self.current_moving_success_rate
        should_keep_best = (
            keep_best_loaded_if_superior
            and (self.best_moving_success_rate >= final_rate + superiority_margin)
        )

        if not should_keep_best:
            self.model.policy.load_state_dict(final_state)
            return False

        return True

    def _render_html_progress(
        self,
        steps: int,
        fps: float,
        eta_sec: float,
        finished: bool = False,
    ) -> str:
        pct = min(100.0, (steps / max(1, self.total_timesteps)) * 100.0)
        succ_rate = self.current_moving_success_rate
        mean_fuel = (
            self._window_fuel_sum / len(self._window_fuels)
            if self._window_fuels
            else 0.0
        )
        bar_color = "#2e7d32" if finished or succ_rate >= 80.0 else "#1976d2"
        status_txt = "Completato" if finished else f"ETA: {eta_sec:.0f}s"

        return (
            f'<div style="font-family:sans-serif; background:#f8f9fa; border:1px solid #dee2e6; '
            f'border-radius:8px; padding:10px 14px; margin:8px 0; max-width:760px;">'
            f'<div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:6px;">'
            f'<span><strong>Addestramento PPO:</strong> {steps:,} / {self.total_timesteps:,} step ({pct:.1f}%)</span>'
            f'<span><strong>{fps:,.0f} step/s</strong> | {status_txt}</span>'
            f'</div>'
            f'<div style="background:#e9ecef; border-radius:6px; height:12px; overflow:hidden;">'
            f'<div style="width:{pct:.1f}%; background:{bar_color}; height:100%; transition:width 0.25s;"></div>'
            f'</div>'
            f'<div style="display:flex; justify-content:space-between; font-size:12px; color:#495057; margin-top:6px;">'
            f'<span>Episodi: {len(self.outcomes)} | Successo recente: <strong>{succ_rate:.1f}%</strong> '
            f'(Picco: {max(0.0, self.best_moving_success_rate):.1f}%)</span>'
            f'<span>Carburante recente: {mean_fuel:.1f} un.</span>'
            f'</div></div>'
        )

    def _update_progress_ui(self, now: float, finished: bool = False) -> None:
        elapsed = max(1e-3, now - self._t_start)
        steps = self.num_timesteps
        fps = steps / elapsed
        remaining_steps = max(0, self.total_timesteps - steps)
        eta_sec = remaining_steps / max(1.0, fps)

        if self._use_html_progress and self._display_handle is not None:
            from IPython.display import HTML
            self._display_handle.update(
                HTML(self._render_html_progress(steps, fps, eta_sec, finished=finished))
            )
        else:
            pct = min(100.0, (steps / max(1, self.total_timesteps)) * 100.0)
            succ_rate = self.current_moving_success_rate
            end_char = "\n" if finished else ""
            print(
                f"\r[play2learn] {pct:5.1f}% | {steps:,}/{self.total_timesteps:,} step | "
                f"{fps:,.0f} step/s | Successo({self.window_size}ep): {succ_rate:5.1f}% | "
                f"ETA: {eta_sec:4.0f}s   ",
                end=end_char,
                flush=True,
            )


def create_ppo_agent(
    env: VecEnv,
    seed: int = 42,
    device: str = "cpu",
    custom_hyperparams: dict[str, Any] | None = None,
    verbose: int = 0,
    use_lr_schedule: bool = True,
) -> PPO:
    """
    Istanzia un agente PPO pre-configurato per convergenza rapida su LunarLander-v3.
    Adatta automaticamente `n_steps` al numero di ambienti paralleli per mantenere
    costante il rollout buffer target (2048 step, multiplo esatto di batch_size=64).
    """
    params = dict(DEFAULT_PPO_PARAMS)
    if custom_hyperparams:
        params.update(custom_hyperparams)

    # Adattamento automatico di n_steps a n_envs se non forzato manualmente
    if not (custom_hyperparams and "n_steps" in custom_hyperparams):
        n_envs = max(1, getattr(env, "num_envs", 4))
        batch_size = int(params.get("batch_size", 64))
        raw_steps = max(batch_size, TARGET_ROLLOUT_SIZE // n_envs)
        params["n_steps"] = max(batch_size, (raw_steps // batch_size) * batch_size)

    # Applicazione learning rate schedule lineare
    if use_lr_schedule and not callable(params.get("learning_rate")):
        lr_start = float(params.get("learning_rate", DEFAULT_LR_START))
        lr_end = min(DEFAULT_LR_END, lr_start * 0.1)
        params["learning_rate"] = LinearSchedule(lr_start, lr_end)

    return PPO(
        env=env,
        seed=seed,
        device=device,
        verbose=verbose,
        **params,
    )
