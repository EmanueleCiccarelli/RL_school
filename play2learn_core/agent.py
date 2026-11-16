"""
Modulo Core di Orchestrazione e Addestramento Agenti PPO (Play2Learn).
Include callback ad alte prestazioni con finestre scorrevoli O(1), snapshot pesi in RAM
e configurazione ottimale di VecEnv per Linux, Windows, macOS e Google Colab.
"""

from __future__ import annotations

import collections
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from tqdm.auto import tqdm


class LinearSchedule:
    """Schedule lineare top-level pienamente serializzabile da pickle/cloudpickle."""

    def __init__(self, initial_value: float, min_value: float = 1e-5):
        self.initial_value = float(initial_value)
        self.min_value = float(min_value)

    def __call__(self, progress_remaining: float) -> float:
        return max(self.min_value, self.initial_value * progress_remaining)


def create_linear_schedule(initial_value: float, min_value: float = 1e-5) -> Callable[[float], float]:
    """Alias retrocompatibile che restituisce un oggetto LinearSchedule serializzabile."""
    return LinearSchedule(initial_value=initial_value, min_value=min_value)


def configure_cpu_threads() -> None:
    """Fissa PyTorch a 1 thread CPU e disattiva la validazione simplex Categorical."""
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    try:
        torch.distributions.Distribution.set_default_validate_args(False)
    except Exception:
        pass


class AdaptiveBatchSizeCallback(BaseCallback):
    """
    Raddoppia il batch_size di PPO (128 -> 256) al 50% di avanzamento per tagliare l'overhead SGD
    tardivo del ~20% e ridurre la varianza della policy vicino alla convergenza.
    """

    def __init__(
        self,
        total_timesteps: int,
        switch_fraction: float = 0.5,
        target_batch_size: int = 256,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.switch_step = int(total_timesteps * switch_fraction)
        self.target_batch_size = int(target_batch_size)
        self._switched = False

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if not self._switched and self.num_timesteps >= self.switch_step:
            rollout_size = getattr(self.model, "n_steps", 256) * getattr(self.model, "n_envs", 8)
            if rollout_size % self.target_batch_size == 0:
                self.model.batch_size = self.target_batch_size
                self._switched = True


@torch.jit.script
def _jit_mlp64_discrete_forward(
    obs: torch.Tensor,
    pi_w0: torch.Tensor, pi_b0: torch.Tensor,
    pi_w1: torch.Tensor, pi_b1: torch.Tensor,
    act_w: torch.Tensor, act_b: torch.Tensor,
    vf_w0: torch.Tensor, vf_b0: torch.Tensor,
    vf_w1: torch.Tensor, vf_b1: torch.Tensor,
    val_w: torch.Tensor, val_b: torch.Tensor,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass stateless TorchScript fused per ActorCritic discreto [64, 64] Tanh.
    Mantiene intatti i moduli nn.Linear standard per compatibilità 100% PPO.save/load.
    """
    h_pi = torch.tanh(F.linear(obs, pi_w0, pi_b0))
    h_pi = torch.tanh(F.linear(h_pi, pi_w1, pi_b1))
    logits = F.linear(h_pi, act_w, act_b)

    h_vf = torch.tanh(F.linear(obs, vf_w0, vf_b0))
    h_vf = torch.tanh(F.linear(h_vf, vf_w1, vf_b1))
    values = F.linear(h_vf, val_w, val_b)

    log_probs = F.log_softmax(logits, dim=-1)
    if deterministic:
        actions = torch.argmax(logits, dim=-1)
    else:
        probs = torch.exp(log_probs)
        actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
    chosen_log_prob = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
    return actions, values, chosen_log_prob


def attach_jit_rollout_forward(model: Any) -> bool:
    """
    Inietta il forward TorchScript stateless durante il rollout su standard SB3 MlpPolicy
    senza alterare lo state_dict o invalidare il salvataggio cloudpickle.
    """
    policy = getattr(model, "policy", None)
    if policy is None:
        return False
    try:
        pi_net = policy.mlp_extractor.policy_net
        vf_net = policy.mlp_extractor.value_net
        if (
            len(pi_net) != 4
            or len(vf_net) != 4
            or not isinstance(pi_net[1], torch.nn.Tanh)
            or not isinstance(vf_net[1], torch.nn.Tanh)
            or policy.features_extractor.__class__.__name__ != "FlattenExtractor"
        ):
            return False

        p_l0, p_l1 = pi_net[0], pi_net[2]
        v_l0, v_l1 = vf_net[0], vf_net[2]
        act_net, val_net = policy.action_net, policy.value_net

        def _fast_forward(obs: torch.Tensor, deterministic: bool = False):
            return _jit_mlp64_discrete_forward(
                obs.float(),
                p_l0.weight, p_l0.bias,
                p_l1.weight, p_l1.bias,
                act_net.weight, act_net.bias,
                v_l0.weight, v_l0.bias,
                v_l1.weight, v_l1.bias,
                val_net.weight, val_net.bias,
                deterministic,
            )

        policy.forward = _fast_forward  # type: ignore[method-assign]
        return True
    except Exception:
        return False


def get_optimal_vec_config(
    n_envs: int | None = None,
    total_timesteps: int = 150_000,
    prefer_subproc: bool | None = None,
) -> tuple[int, bool]:
    """Seleziona DummyVecEnv su Windows/macOS/Colab (<=4 thread) e SubprocVecEnv su Linux >=6 core."""
    cpu_count = os.cpu_count() or 2
    is_linux = sys.platform.startswith("linux")

    if prefer_subproc is not None:
        use_subproc = prefer_subproc
    else:
        use_subproc = is_linux and (cpu_count >= 6) and (total_timesteps >= 200_000)

    resolved_envs = 8 if n_envs is None else max(1, int(n_envs))
    if resolved_envs == 1:
        use_subproc = False
    return resolved_envs, use_subproc


class BaseRewardTrackerCallback(BaseCallback):
    """
    Callback didattica O(1) con snapshot pesi in RAM (zero disk I/O bloccante durante il training)
    e cattura milestone sicura tramite clone() dello state_dict della policy.
    """

    def __init__(
        self,
        total_timesteps: int,
        show_progress: bool = True,
        save_checkpoints: bool = True,
        checkpoint_dir: Path | str | None = None,
        save_freq: int = 10_000,
        window_size: int = 50,
        milestone_callback: Callable[[int, Any], None] | None = None,
        milestone_freq: int = 30_000,
        verbose: int = 0,
        metric_name: str = "fuel",
    ):
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self.show_progress = show_progress
        self.save_checkpoints = save_checkpoints
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.save_freq = save_freq
        self.window_size = window_size
        self.milestone_callback = milestone_callback
        self.milestone_freq = max(1, int(milestone_freq if milestone_freq != 30_000 else max(10_000, total_timesteps // 5)))
        self.metric_name = metric_name

        self.episode_rewards: collections.deque[float] = collections.deque(maxlen=window_size)
        self.episode_lengths: collections.deque[int] = collections.deque(maxlen=window_size)
        self.episode_successes: collections.deque[int] = collections.deque(maxlen=window_size)
        self.episode_secondary: collections.deque[float] = collections.deque(maxlen=window_size)

        self._window_reward_sum = 0.0
        self._window_secondary_sum = 0.0
        self._window_success_sum = 0

        self.total_episodes = 0
        self.successes = 0
        self.crashes = 0
        self.timeouts = 0
        self.best_mean_reward = -float("inf")
        self.best_success_rate = -1.0
        self.best_secondary = -float("inf")
        self._best_policy_state: dict[str, torch.Tensor] | None = None
        self.best_model_path: Path | None = None

        self.milestone_snapshots: list[tuple[int, dict[str, torch.Tensor]]] = []
        self._next_milestone_step = self.milestone_freq

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
        n = len(self.episode_successes)
        return (self._window_success_sum / n) * 100.0 if n > 0 else 0.0

    def _clone_policy_state(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.policy.state_dict().items()}

    def _on_training_start(self) -> None:
        self.start_time = time.perf_counter()
        self._last_ui_update_time = self.start_time
        if hasattr(self, "model") and hasattr(self.model, "policy"):
            self.milestone_snapshots.append((0, self._clone_policy_state()))
        if self.show_progress:
            self.pbar = tqdm(
                total=self.total_timesteps,
                desc="Addestramento Agente RL",
                unit="step",
                dynamic_ncols=True,
            )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue
            ep_info = info.get("episode")
            if ep_info is None:
                continue

            self.total_episodes += 1
            r = float(ep_info.get("r", 0.0))
            l = int(ep_info.get("l", 0))

            if len(self.episode_rewards) == self.window_size:
                self._window_reward_sum -= self.episode_rewards[0]
            self.episode_rewards.append(r)
            self._window_reward_sum += r
            self.episode_lengths.append(l)

            succ_int = 1 if info.get("is_success", False) else 0
            if len(self.episode_successes) == self.window_size:
                self._window_success_sum -= self.episode_successes[0]
            self.episode_successes.append(succ_int)
            self._window_success_sum += succ_int

            if succ_int:
                self.successes += 1
            elif info.get("is_timeout", False):
                self.timeouts += 1
            else:
                self.crashes += 1

            telemetry = info.get("telemetry", {})
            sec_val = float(
                telemetry.get(self.metric_name, 0.0)
                if isinstance(telemetry, dict)
                else info.get(self.metric_name, 0.0)
            )
            if len(self.episode_secondary) == self.window_size:
                self._window_secondary_sum -= self.episode_secondary[0]
            self.episode_secondary.append(sec_val)
            self._window_secondary_sum += sec_val

            # Snapshot in RAM del miglior modello
            if self.save_checkpoints and len(self.episode_rewards) >= min(10, self.window_size):
                rate = self.current_moving_success_rate
                rew = self.current_moving_reward
                sec = self.current_moving_secondary
                is_better = (
                    (rate > self.best_success_rate + 2.0)
                    or (abs(rate - self.best_success_rate) <= 2.0 and sec > self.best_secondary + 15.0)
                    or (abs(rate - self.best_success_rate) <= 2.0 and rew > self.best_mean_reward + 10.0)
                )
                if is_better:
                    self.best_success_rate = rate
                    self.best_secondary = sec
                    self.best_mean_reward = rew
                    self._best_policy_state = self._clone_policy_state()

        if self.num_timesteps >= self._next_milestone_step and len(self.milestone_snapshots) < 6:
            snap = self._clone_policy_state()
            self.milestone_snapshots.append((self.num_timesteps, snap))
            if self.milestone_callback:
                try:
                    self.milestone_callback(self.num_timesteps, snap)
                except Exception:
                    pass
            self._next_milestone_step += self.milestone_freq

        if self.pbar and (self.n_calls & 63) == 0:
            now = time.perf_counter()
            if now - self._last_ui_update_time >= 0.35:
                self._last_ui_update_time = now
                self.pbar.n = min(self.num_timesteps, self.total_timesteps)
                self.pbar.set_postfix({
                    "Reward": f"{self.current_moving_reward:.1f}",
                    "Successi": f"{self.current_moving_success_rate:.0f}%",
                    self.metric_name.capitalize(): f"{self.current_moving_secondary:.1f}",
                })

        return True

    def flush_best_model_to_disk(self) -> bool:
        """Scrive su disco una sola volta il miglior modello catturato in RAM."""
        if not self.checkpoint_dir or self._best_policy_state is None:
            return False
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        final_state = self._clone_policy_state()
        best_path = self.checkpoint_dir / "best_model.zip"
        try:
            self.model.policy.load_state_dict(self._best_policy_state)
            self.model.save(str(best_path))
            self.best_model_path = best_path
        finally:
            self.model.policy.load_state_dict(final_state)
        return True

    def _on_training_end(self) -> None:
        if hasattr(self, "model") and hasattr(self.model, "policy"):
            self.milestone_snapshots.append((max(1, self.num_timesteps), self._clone_policy_state()))
        if self.pbar:
            self.pbar.n = self.total_timesteps
            self.pbar.refresh()
            self.pbar.close()
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.flush_best_model_to_disk()
            self.model.save(str(self.checkpoint_dir / "final_model.zip"))


def make_parallel_envs(
    env_fns: list[Callable[[], Any]],
    use_subproc: bool = False,
    vecnorm_kwargs: dict[str, Any] | None = None,
) -> VecNormalize:
    """Crea e normalizza in modo vettorizzato gli ambienti Gymnasium con fallback sicuro."""
    if use_subproc and len(env_fns) > 1:
        try:
            start_method = "forkserver" if sys.platform.startswith("linux") else "spawn"
            vec_env = SubprocVecEnv(env_fns, start_method=start_method)
        except Exception:
            vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    params = {
        "norm_obs": False,
        "norm_reward": True,
        "clip_reward": 10.0,
        "gamma": 0.995,
    }
    if vecnorm_kwargs:
        params.update(vecnorm_kwargs)
    return VecNormalize(vec_env, **params)
