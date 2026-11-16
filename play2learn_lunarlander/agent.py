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
    class _DummyBaseCallback:
        def __init__(self, verbose: int = 0):
            self.verbose = verbose
    BaseCallback = _DummyBaseCallback  # type: ignore
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
    - Mostra una progress bar HTML/console throttled (ogni 1.5s) senza lag su Colab.
    """

    def __init__(
        self,
        best_model_save_dir: Path | str | None = None,
        total_timesteps: int = 0,
        show_progress: bool = True,
        progress_interval_sec: float = 1.0,
        window_size: int = 30,
        capture_milestones: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.best_model_save_dir = Path(best_model_save_dir) if best_model_save_dir else None
        self.total_timesteps = int(total_timesteps)
        self.show_progress = show_progress
        self.progress_interval_sec = float(progress_interval_sec)
        self.window_size = int(window_size)
        self.capture_milestones = capture_milestones

        # Snapshot intermedi per timelapse di evoluzione apprendimento (RAM, overhead 0)
        self.milestone_snapshots: list[dict[str, Any]] = []
        self._milestone_targets: list[float] = [0.10, 0.35, 0.65]
        self._next_milestone_idx: int = 0

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

        # Finestra scorrevole O(1) con somme cumulate per azzerare l'overhead di calcolo
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
            if is_succ:
                self.outcomes.append("success")
                self.successes += 1
                succ_flag = 1
            elif info.get("is_crashed", False):
                self.outcomes.append("crash")
                self.crashes += 1
                succ_flag = 0
            elif info.get("is_timeout", False):
                self.outcomes.append("timeout")
                self.timeouts += 1
                succ_flag = 0
            else:
                self.outcomes.append("unknown")
                succ_flag = 0

            tel = info.get("telemetry")
            if tel:
                fuel = float(tel["fuel_used"])
                self.fuel_used_list.append(fuel)
                self.air_steps_list.append(int(tel["air_steps"]))
                self.ground_steps_list.append(int(tel["ground_steps"]))
                self.final_x_list.append(float(tel["final_abs_x"]))
                self.final_vy_list.append(float(tel["final_abs_vy"]))
                self.final_angle_list.append(float(tel["final_abs_angle"]))
            else:
                fuel = 0.0
                self.fuel_used_list.append(0.0)
                self.air_steps_list.append(ep_l)
                self.ground_steps_list.append(0)
                self.final_x_list.append(0.0)
                self.final_vy_list.append(0.0)
                self.final_angle_list.append(0.0)

            # Aggiornamento O(1) puro della finestra scorrevole e delle somme
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

            # Snapshot in RAM del miglior modello (zero disk I/O!)
            if self.best_model_save_dir and len(self._window_outcomes) == self.window_size:
                curr_rate = (self._window_successes / self.window_size) * 100.0
                curr_fuel = self._window_fuel_sum / self.window_size
                is_better = (
                    curr_rate >= 40.0
                    and (
                        curr_rate > self.best_moving_success_rate
                        or (
                            curr_rate == self.best_moving_success_rate
                            and curr_fuel < self._best_moving_fuel - 2.0
                        )
                    )
                )
                if is_better:
                    self.best_moving_success_rate = curr_rate
                    self._best_moving_fuel = curr_fuel
                    self._best_policy_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.policy.state_dict().items()
                    }

        # Snapshot intermedi per timelapse evoluzione (RAM, eseguito 3 volte durante tutto il training)
        if self.capture_milestones and self.total_timesteps > 0 and self._next_milestone_idx < len(self._milestone_targets):
            target_step = int(self._milestone_targets[self._next_milestone_idx] * self.total_timesteps)
            if self.num_timesteps >= target_step:
                pct = int(self._milestone_targets[self._next_milestone_idx] * 100)
                self.milestone_snapshots.append({
                    "step": self.num_timesteps,
                    "percent": pct,
                    "success_rate": self.current_moving_success_rate,
                    "state_dict": {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.policy.state_dict().items()
                    },
                })
                self._next_milestone_idx += 1

        # Aggiornamento UI throttled (controllo temporale ogni 64 step per azzerare chiamate di sistema perf_counter)
        if self.show_progress and self.total_timesteps > 0 and (self.n_calls & 63) == 0:
            now = time.perf_counter()
            if (now - self._last_ui_update) >= self.progress_interval_sec:
                self._last_ui_update = now
                self._update_progress_ui(now)

        return True

    def _on_training_end(self) -> None:
        if self.capture_milestones and hasattr(self, "model") and hasattr(self.model, "policy"):
            self.milestone_snapshots.append({
                "step": max(1, self.num_timesteps),
                "percent": 100,
                "success_rate": self.current_moving_success_rate,
                "state_dict": {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.policy.state_dict().items()
                },
            })

        if self.show_progress and self.total_timesteps > 0:
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
        superiority_margin: float = 10.0,
    ) -> bool:
        """
        Scrive su disco una sola volta il miglior modello catturato in RAM.
        Se `keep_best_loaded_if_superior=True` e il picco in training supera la
        media mobile finale di almeno `superiority_margin` punti percentuali,
        mantiene i pesi migliori caricati nell'agente per la valutazione.

        Restituisce True se i pesi migliori sono rimasti attivi in `self.model`.
        """
        if self._best_policy_state is None or self.best_model_save_dir is None:
            return False

        self.best_model_save_dir.mkdir(parents=True, exist_ok=True)
        save_p = self.best_model_save_dir / "best_model.zip"

        final_state = {
            k: v.detach().cpu().clone()
            for k, v in self.model.policy.state_dict().items()
        }

        # Carica pesi migliori e salva una sola volta su disco
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
        mean_reward = self.current_moving_reward
        mean_fuel = self.current_moving_fuel
        bar_color = "#2e7d32" if finished or succ_rate >= 80.0 else "#1976d2"
        status_txt = "Completato" if finished else f"ETA: {eta_sec:.0f}s"
        n_fails = self.crashes + self.timeouts

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
            f'<span>Episodi: <strong>{len(self.outcomes)}</strong> (<strong>{self.successes}</strong> 🟢 | <strong>{n_fails}</strong> 🔴)</span>'
            f'<span>Successo recente ({self.window_size} ep): <strong style="color:{bar_color}; font-size:13px;">{succ_rate:.1f}%</strong> '
            f'(Picco: {max(0.0, self.best_moving_success_rate):.1f}%)</span>'
            f'</div>'
            f'<div style="display:flex; justify-content:space-between; font-size:12px; color:#495057; margin-top:4px;">'
            f'<span>Current Reward ({self.window_size} ep): <strong style="color:#1565c0; font-size:13px;">{mean_reward:+.1f}</strong></span>'
            f'<span>Carburante recente ({self.window_size} ep): <strong>{mean_fuel:.1f}</strong> un.</span>'
            f'</div>'
            f'</div>'
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
            mean_reward = self.current_moving_reward
            mean_fuel = self.current_moving_fuel
            end_char = "\n" if finished else ""
            print(
                f"\r[play2learn] {pct:5.1f}% | {steps:,}/{self.total_timesteps:,} step | "
                f"{fps:,.0f} st/s | Reward: {mean_reward:+7.1f} | "
                f"Succ: {succ_rate:5.1f}% ({self.successes} succ) | "
                f"Carb: {mean_fuel:5.1f} | ETA: {eta_sec:4.0f}s   ",
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
        # Allinea al multiplo di batch_size per evitare minibatch troncati
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
