"""
Modulo Core di Orchestrazione e Addestramento Agenti PPO (Play2Learn).
Include callback ad alte prestazioni con finestre scorrevoli O(1) e gestione sicura di VecEnv.
"""

from __future__ import annotations

import collections
import time
from pathlib import Path
from typing import Any, Callable

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from tqdm.auto import tqdm


def create_linear_schedule(initial_value: float, min_value: float = 1e-5) -> Callable[[float], float]:
    """Crea uno scheduler di learning rate lineare decrescente da initial_value a min_value."""
    def schedule(progress_remaining: float) -> float:
        return max(min_value, initial_value * progress_remaining)
    return schedule


class BaseRewardTrackerCallback(BaseCallback):
    """
    Callback didattica ad alta efficienza per il monitoraggio dell'addestramento PPO.
    Mantiene somme scorrevoli per calcoli O(1) di medie e success rate.
    """

    def __init__(
        self,
        total_timesteps: int,
        show_progress: bool = True,
        save_checkpoints: bool = True,
        checkpoint_dir: Path | str | None = None,
        save_freq: int = 10_000,
        window_size: int = 100,
        milestone_callback: Callable[[int, Any], None] | None = None,
        milestone_freq: int = 40_000,
        verbose: int = 0,
        metric_name: str = "fuel",
    ):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.show_progress = show_progress
        self.save_checkpoints = save_checkpoints
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.save_freq = save_freq
        self.window_size = window_size
        self.milestone_callback = milestone_callback
        self.milestone_freq = milestone_freq
        self.metric_name = metric_name

        # Finestre scorrevoli O(1)
        self.episode_rewards: collections.deque[float] = collections.deque(maxlen=window_size)
        self.episode_lengths: collections.deque[int] = collections.deque(maxlen=window_size)
        self.episode_successes: collections.deque[bool] = collections.deque(maxlen=window_size)
        self.episode_secondary: collections.deque[float] = collections.deque(maxlen=window_size)

        # Somme scorrevoli per media O(1)
        self._window_reward_sum = 0.0
        self._window_secondary_sum = 0.0

        # Contatori cumulativi
        self.total_episodes = 0
        self.successes = 0
        self.crashes = 0
        self.timeouts = 0
        self.best_mean_reward = -float("inf")
        self.best_success_rate = -1.0
        self.pbar: tqdm | None = None
        self.start_time = 0.0
        self._last_ui_update_time = 0.0

    @property
    def current_moving_reward(self) -> float:
        n = len(self.episode_rewards)
        return (self._window_reward_sum / n) if n > 0 else 0.0

    @property
    def current_moving_secondary(self) -> float:
        n = len(self.episode_secondary)
        return (self._window_secondary_sum / n) if n > 0 else 0.0

    @property
    def current_moving_success_rate(self) -> float:
        if not self.episode_successes:
            return 0.0
        return (sum(self.episode_successes) / len(self.episode_successes)) * 100.0

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self._last_ui_update_time = self.start_time
        if self.show_progress:
            self.pbar = tqdm(
                total=self.total_timesteps,
                desc="🚀 Addestramento Agente RL",
                unit="step",
                dynamic_ncols=True,
            )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue

            ep_info = info.get("episode")
            if ep_info is not None:
                self.total_episodes += 1
                r = float(ep_info.get("r", 0.0))
                l = int(ep_info.get("l", 0))

                # Aggiornamento finestra reward O(1)
                if len(self.episode_rewards) == self.window_size:
                    self._window_reward_sum -= self.episode_rewards[0]
                self.episode_rewards.append(r)
                self._window_reward_sum += r
                self.episode_lengths.append(l)

                succ = bool(info.get("is_success", False))
                self.episode_successes.append(succ)
                if succ:
                    self.successes += 1
                elif info.get("is_timeout", False):
                    self.timeouts += 1
                else:
                    self.crashes += 1

                # Misura secondaria (es. carburante o distanza o velocità)
                sec_val = 0.0
                telemetry = info.get("telemetry", {})
                if isinstance(telemetry, dict):
                    sec_val = float(telemetry.get(self.metric_name, 0.0))
                else:
                    sec_val = float(info.get(self.metric_name, 0.0))

                if len(self.episode_secondary) == self.window_size:
                    self._window_secondary_sum -= self.episode_secondary[0]
                self.episode_secondary.append(sec_val)
                self._window_secondary_sum += sec_val

        # Salvataggio milestone periodica
        if self.milestone_callback and (self.num_timesteps % self.milestone_freq < self.training_env.num_envs):
            try:
                self.milestone_callback(self.num_timesteps, self.model)
            except Exception:
                pass

        # Salvataggio checkpoint
        if self.save_checkpoints and self.checkpoint_dir and (self.num_timesteps % self.save_freq < self.training_env.num_envs):
            self.flush_best_model_to_disk()

        # Aggiornamento progress bar throttled (solo ogni 64 step o 0.25s)
        if self.pbar:
            self.pbar.n = self.num_timesteps
            if (self.n_calls & 63) == 0:
                now = time.time()
                if now - self._last_ui_update_time >= 0.25:
                    self._last_ui_update_time = now
                    sec_val = self.current_moving_secondary
                    self.pbar.set_postfix({
                        "Reward": f"{self.current_moving_reward:.1f}",
                        "Successi": f"{self.current_moving_success_rate:.0f}%",
                        self.metric_name.capitalize(): f"{sec_val:.1f}",
                    })

        return True

    def flush_best_model_to_disk(self) -> None:
        """Salva il modello migliore su disco in modo sicuro."""
        if not self.checkpoint_dir:
            return

        rate = self.current_moving_success_rate
        rew = self.current_moving_reward

        is_better = (rate > self.best_success_rate + 2.0) or (
            abs(rate - self.best_success_rate) <= 2.0 and rew > self.best_mean_reward + 10.0
        )

        if is_better and rate >= 40.0:
            self.best_success_rate = rate
            self.best_mean_reward = rew
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.checkpoint_dir / "best_model.zip"))

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.n = self.total_timesteps
            self.pbar.close()
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.checkpoint_dir / "final_model.zip"))


def make_parallel_envs(
    env_fns: list[Callable[[], Any]],
    use_subproc: bool = False,
) -> VecNormalize:
    """Crea e normalizza in modo vettorizzato gli ambienti Gymnasium."""
    if use_subproc and len(env_fns) > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    # norm_obs=False preserva le osservazioni raw fisiche per l'inferenza deterministica
    norm_env = VecNormalize(
        vec_env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
        gamma=0.995,
    )
    return norm_env
